import re
from typing import Dict, List, Optional, Set, Tuple

from .types import (
    ErrorAnalysis,
    OrderedRefinementContext,
    RefinementInput,
    SingleErrorContext,
)
from .types import ERROR_NEEDS_SUBGRAPH, ERROR_PRIORITY, SUBTREE_SCOPE
from .prompts import format_few_shot_for_prompt


class ContextBuilder:

    def build(
        self,
        refinement_input: RefinementInput,
        error_analyses: List[ErrorAnalysis]
    ) -> OrderedRefinementContext:
        error_contexts = []

        for analysis in error_analyses:
            error_type = analysis.error_type

            subtree = self._extract_subtree(
                refinement_input.ast,
                analysis.located_nodes,
                error_type
            )

            subgraph = None
            if ERROR_NEEDS_SUBGRAPH.get(error_type, False):
                subgraph = self._extract_subgraph(
                    refinement_input.schema_structure_nodes,
                    refinement_input.schema_structure_edges,
                    analysis.involved_elements,
                    error_type
                )

            few_shot = format_few_shot_for_prompt(error_type)
            priority = ERROR_PRIORITY.get(error_type, 99)

            error_contexts.append(SingleErrorContext(
                error_type=error_type,
                priority=priority,
                subtree=subtree,
                subgraph=subgraph,
                filled_guideline=analysis.filled_guideline,
                few_shot=few_shot
            ))

        error_contexts.sort(key=lambda x: x.priority)

        return OrderedRefinementContext(
            full_sql=refinement_input.predicted_sql,
            error_contexts=error_contexts
        )

    def _extract_subtree(
        self,
        ast_text: str,
        located_nodes: List[str],
        error_type: str
    ) -> str:
        if not ast_text or not located_nodes:
            scope = SUBTREE_SCOPE.get(error_type, "unknown")
            return f"(No specific nodes located - error affects {scope})"

        ast_lines = ast_text.strip().split("\n")
        parsed_nodes = []

        for line in ast_lines:
            match = re.match(r'(Node\[\d+\])\s*\|\s*([^\|]+)\s*(?:\|\s*(\w+))?\s*(?:\|\s*(.+))?', line.strip())
            if match:
                node_id = match.group(1)
                path = match.group(2).strip()
                node_type = match.group(3).strip() if match.group(3) else ""
                value = match.group(4).strip() if match.group(4) else ""
                parsed_nodes.append({
                    "id": node_id,
                    "path": path,
                    "type": node_type,
                    "value": value,
                    "line": line.strip()
                })

        located_indices = set()
        for node_ref in located_nodes:
            match = re.search(r'Node\[(\d+)\]', node_ref)
            if match:
                located_indices.add(int(match.group(1)))

        scope = SUBTREE_SCOPE.get(error_type, "expression")
        relevant_nodes = self._find_scope_nodes(parsed_nodes, located_indices, scope)

        output_lines = []
        for node in relevant_nodes:
            line = node["line"]
            match = re.search(r'Node\[(\d+)\]', node["id"])
            if match and int(match.group(1)) in located_indices:
                line += "  <- [ERROR]"
            output_lines.append(line)

        return "\n".join(output_lines) if output_lines else "(No subtree extracted)"

    def _find_scope_nodes(
        self,
        parsed_nodes: List[Dict],
        located_indices: Set[int],
        scope: str
    ) -> List[Dict]:
        if not located_indices or not parsed_nodes:
            return parsed_nodes[:10]

        located_paths = []
        for node in parsed_nodes:
            match = re.search(r'Node\[(\d+)\]', node["id"])
            if match and int(match.group(1)) in located_indices:
                located_paths.append(node["path"])

        if not located_paths:
            return parsed_nodes[:10]

        scope_prefixes = self._get_scope_prefixes(located_paths, scope)

        relevant = []
        for node in parsed_nodes:
            path = node["path"]
            for prefix in scope_prefixes:
                if path.startswith(prefix) or prefix.startswith(path):
                    relevant.append(node)
                    break

        return relevant if relevant else parsed_nodes[:10]

    def _get_scope_prefixes(self, paths: List[str], scope: str) -> List[str]:
        prefixes = set()

        for path in paths:
            parts = path.split(".")

            if scope == "select_clause":
                prefixes.add("select")
            elif scope == "from_clause":
                prefixes.add("select.from")
            elif scope == "where_clause":
                prefixes.add("select.where")
            elif scope == "join_clause":
                for i, part in enumerate(parts):
                    if part == "join":
                        prefixes.add(".".join(parts[:i+1]))
                        break
                else:
                    prefixes.add("select.join")
            elif scope == "condition":
                for i, part in enumerate(parts):
                    if part in ("eq", "neq", "lt", "gt", "lte", "gte", "like", "in", "between", "and", "or"):
                        prefixes.add(".".join(parts[:i+1]))
                        break
                else:
                    prefixes.add(path)
            elif scope == "expression":
                if len(parts) > 1:
                    prefixes.add(".".join(parts[:-1]))
                else:
                    prefixes.add(path)
            elif scope == "function":
                for i, part in enumerate(parts):
                    if part in ("count", "sum", "avg", "max", "min", "case", "strftime", "substr"):
                        prefixes.add(".".join(parts[:i+1]))
                        break
                else:
                    prefixes.add(path)
            elif scope == "clause":
                for clause in ("group", "order", "limit", "having"):
                    if clause in parts:
                        idx = parts.index(clause)
                        prefixes.add(".".join(parts[:idx+1]))
                        break
                else:
                    prefixes.add(path)
            else:
                prefixes.add(path)

        return list(prefixes)

    def _extract_subgraph(
        self,
        nodes_text: str,
        edges_text: str,
        involved_elements: List[str],
        error_type: str
    ) -> str:
        if not involved_elements:
            return "Not applicable - no specific elements identified."

        tables = set()
        columns = set()

        for elem in involved_elements:
            if "." in elem:
                table, col = elem.split(".", 1)
                tables.add(table.lower())
                columns.add(elem.lower())
            else:
                tables.add(elem.lower())

        result_parts = []

        if columns:
            result_parts.append("**Relevant Columns**:")
            for line in nodes_text.split("\n"):
                line_lower = line.lower()
                for col in columns:
                    if col in line_lower:
                        result_parts.append(f"- {line.strip()}")
                        break

        if tables:
            result_parts.append("\n**Relevant Tables**:")
            for table in tables:
                result_parts.append(f"- {table}")

        if error_type in ("Table Missing", "Table Mismatch", "Attribute Mismatch"):
            fk_edges = []
            for edge in edges_text.split(";"):
                if "foreign_key_to" in edge.lower():
                    for table in tables:
                        if table in edge.lower():
                            fk_edges.append(edge.strip())
                            break

            if fk_edges:
                result_parts.append("\n**Foreign Key Relationships**:")
                for edge in fk_edges:
                    result_parts.append(f"- {edge}")

        if error_type in ("Attribute Missing", "Attribute Mismatch", "Condition Missing"):
            nl_edges = []
            for edge in edges_text.split(";"):
                if "relates_to_column" in edge.lower() or "relates_to_table" in edge.lower():
                    nl_edges.append(edge.strip())

            if nl_edges:
                result_parts.append("\n**Question-Schema Mappings**:")
                for edge in nl_edges[:5]:
                    result_parts.append(f"- {edge}")

        return "\n".join(result_parts) if result_parts else "No relevant subgraph extracted."

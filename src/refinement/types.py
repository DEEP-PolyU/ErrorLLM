import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


ERROR_TOKEN_TO_NAME: Dict[str, str] = {
    "<error_1>": "Attribute Mismatch",
    "<error_2>": "Attribute Redundancy",
    "<error_3>": "Attribute Missing",
    "<error_4>": "Table Mismatch",
    "<error_5>": "Table Redundancy",
    "<error_6>": "Table Missing",
    "<error_7>": "Value Mismatch",
    "<error_8>": "Condition Missing",
    "<error_9>": "Condition Error",
    "<error_10>": "Function Error",
    "<error_11>": "Clause Error",
    "<error_12>": "Modifier Error",
}

ERROR_NAME_TO_TOKEN: Dict[str, str] = {v: k for k, v in ERROR_TOKEN_TO_NAME.items()}

ERROR_DESCRIPTIONS: Dict[str, str] = {
    "Attribute Mismatch": "The attribute [A] may be wrong.",
    "Attribute Redundancy": "The attribute [A] may not be mentioned in the NLQ.",
    "Attribute Missing": "The attribute [A] may be missing.",
    "Table Mismatch": "The table [T] may be wrong.",
    "Table Redundancy": "The table [T] may be unnecessary.",
    "Table Missing": "The table [T] may be missing.",
    "Value Mismatch": "The value [V] in condition [C] may be wrong; Data Format Mismatch: The data format of value [V] in attribute [A] may be wrong.",
    "Condition Missing": "The condition [C] in NLQ may be missing; The SQL fails to include implicit conditions [C] (e.g., IS NOT NULL).",
    "Condition Error": "The condition [C] may be wrong; The condition [C] which not mentioned in NLQ.",
    "Function Error": "The usage of aggregate functions [F] (e.g., SUM, AVG) is incorrect; The usage of date/time functions [F] (e.g., JULIANDAY, strftime) is incorrect; The usage of string functions [F] (e.g., SUBSTR) is incorrect; The usage of conditional functions [F] (e.g., IIF, CASE WHEN) is incorrect.",
    "Clause Error": "The clause [K] (e.g., GROUP BY) is missing; The clause [K] (e.g., GROUP BY) is redundant.",
    "Modifier Error": "The usage of DISTINCT, DESC/ASC is either omitted or incorrectly applied.",
}

ERROR_PRIORITY: Dict[str, int] = {
    "Table Missing": 1,
    "Table Mismatch": 2,
    "Table Redundancy": 2,
    "Clause Error": 2,
    "Attribute Missing": 3,
    "Condition Missing": 3,
    "Attribute Mismatch": 4,
    "Condition Error": 4,
    "Attribute Redundancy": 5,
    "Value Mismatch": 5,
    "Function Error": 5,
    "Modifier Error": 5,
}

ERROR_NEEDS_SUBGRAPH: Dict[str, bool] = {
    "Attribute Mismatch": True,
    "Attribute Redundancy": False,
    "Attribute Missing": True,
    "Table Mismatch": True,
    "Table Redundancy": False,
    "Table Missing": True,
    "Value Mismatch": True,
    "Condition Missing": True,
    "Condition Error": False,
    "Function Error": False,
    "Clause Error": False,
    "Modifier Error": False,
}

SUBTREE_SCOPE: Dict[str, str] = {
    "Attribute Mismatch": "expression",
    "Attribute Redundancy": "select_item",
    "Attribute Missing": "select_clause",
    "Table Mismatch": "table_ref",
    "Table Redundancy": "join_clause",
    "Table Missing": "from_clause",
    "Value Mismatch": "condition",
    "Condition Missing": "where_clause",
    "Condition Error": "condition",
    "Function Error": "function",
    "Clause Error": "clause",
    "Modifier Error": "select_clause",
}


def parse_error_tokens(output_str: str) -> List[str]:
    if output_str.strip() == "<no_error>":
        return []
    pattern = r"<error_\d+>"
    return re.findall(pattern, output_str)


def get_natural_language_errors(output_str: str) -> List[Tuple[str, str, str]]:
    tokens = parse_error_tokens(output_str)
    errors = []
    for token in tokens:
        if token in ERROR_TOKEN_TO_NAME:
            name = ERROR_TOKEN_TO_NAME[token]
            desc = ERROR_DESCRIPTIONS.get(name, "")
            priority = ERROR_PRIORITY.get(name, 99)
            errors.append((token, name, desc, priority))
    errors.sort(key=lambda x: x[3])
    return [(t, n, d) for t, n, d, _ in errors]


def format_errors_for_display(output_str: str) -> str:
    errors = get_natural_language_errors(output_str)
    if not errors:
        return "No errors detected."
    lines = []
    for i, (token, name, desc) in enumerate(errors, 1):
        lines.append(f"{i}. {name}: {desc}")
    return "\n".join(lines)


@dataclass
class RefinementInput:
    question: str
    external_knowledge: str
    schema_structure_nodes: str
    schema_structure_edges: str
    predicted_sql: str
    ast: str
    error_types: List[str]
    rule_based_results: Optional[str] = None
    execution_results: Optional[str] = None


@dataclass
class ErrorAnalysis:
    error_type: str
    located_nodes: List[str]
    involved_elements: List[str]
    filled_guideline: Dict[str, Any]


@dataclass
class SingleErrorContext:
    error_type: str
    priority: int
    subtree: str
    subgraph: Optional[str]
    filled_guideline: Dict[str, Any]
    few_shot: str


@dataclass
class OrderedRefinementContext:
    full_sql: str
    error_contexts: List[SingleErrorContext]


@dataclass
class RefinementResult:
    original_sql: str
    refined_sql: str
    error_analyses: List[ErrorAnalysis] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None

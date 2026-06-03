import random
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Set

import sqlglot
from sqlglot import exp

INVALID_SQLITE_FUNCTIONS = [
    "YEAR", "MONTH", "DAY", "DATEDIFF", "NOW", "CURDATE", "GETDATE",
    "DATEADD", "DATENAME", "DATEPART", "ISNULL", "NVL", "CONVERT",
]

_OPERATOR_SWAPS = {
    exp.EQ: exp.GT,
    exp.GT: exp.LT,
    exp.LT: exp.GTE,
    exp.GTE: exp.LTE,
    exp.LTE: exp.NEQ,
    exp.NEQ: exp.EQ,
}


def _norm(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_").replace("`", "").replace('"', "").strip()


def _get_table_columns(db_path: str):
    table_cols = {}
    try:
        conn = sqlite3.connect(db_path, timeout=1.0)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for row in cursor.fetchall():
            t = row[0]
            cursor.execute(f'PRAGMA table_info("{t}")')
            table_cols[t.lower()] = [r[1] for r in cursor.fetchall()]
        conn.close()
    except Exception:
        pass
    return table_cols


@dataclass
class PerturbationResult:
    perturbed_sql: str
    error_type: str
    description: str


class PerturbationOperator(ABC):
    error_type: str = ""

    @abstractmethod
    def inject(
        self,
        parsed_ast: exp.Expression,
        db_tables: Set[str],
        db_columns: Set[str],
        db_path: str,
    ) -> Optional[PerturbationResult]:
        ...

    def _to_sql(self, ast: exp.Expression) -> str:
        return ast.sql(dialect="sqlite")


class AttributeRedundancyOperator(PerturbationOperator):
    error_type = "attribute_redundancy"

    def inject(self, parsed_ast, db_tables, db_columns, db_path):
        ast = parsed_ast.copy()
        columns = list(ast.find_all(exp.Column))
        if not columns:
            return None

        table_cols = _get_table_columns(db_path)
        all_real_columns = []
        for cols in table_cols.values():
            all_real_columns.extend(cols)
        if not all_real_columns:
            return None

        random.shuffle(columns)
        for col_node in columns:
            original_name = col_node.name
            if not original_name:
                continue
            candidates = [c for c in all_real_columns if _norm(c) != _norm(original_name)]
            if not candidates:
                continue
            replacement = random.choice(candidates)
            col_node.set("this", exp.to_identifier(replacement))
            sql = self._to_sql(ast)
            return PerturbationResult(
                perturbed_sql=sql,
                error_type=self.error_type,
                description=f"Replaced column '{original_name}' with unrelated schema column '{replacement}'",
            )
        return None


class AttributeMissingOperator(PerturbationOperator):
    error_type = "attribute_missing"

    def inject(self, parsed_ast, db_tables, db_columns, db_path):
        ast = parsed_ast.copy()
        select = ast.find(exp.Select)
        if select is None:
            return None
        expressions = select.args.get("expressions")
        if not expressions or len(expressions) < 2:
            return None
        idx = random.randrange(len(expressions))
        removed = expressions.pop(idx)
        desc = str(removed)
        sql = self._to_sql(ast)
        return PerturbationResult(
            perturbed_sql=sql,
            error_type=self.error_type,
            description=f"Removed SELECT expression: {desc}",
        )


class AttributeMismatchOperator(PerturbationOperator):
    error_type = "attribute_mismatch"

    def inject(self, parsed_ast, db_tables, db_columns, db_path):
        ast = parsed_ast.copy()
        columns = list(ast.find_all(exp.Column))
        if not columns:
            return None

        table_cols = _get_table_columns(db_path)

        if len(table_cols) < 2:
            return None

        random.shuffle(columns)
        for col_node in columns:
            original_name = col_node.name
            if not original_name:
                continue
            original_norm = _norm(original_name)

            source_table = None
            for t, cols in table_cols.items():
                if original_norm in [_norm(c) for c in cols]:
                    source_table = t
                    break
            if source_table is None:
                continue

            other_tables = [t for t in table_cols if t != source_table and table_cols[t]]
            if not other_tables:
                continue
            other_table = random.choice(other_tables)
            replacement = random.choice(table_cols[other_table])
            if _norm(replacement) == original_norm:
                continue
            col_node.set("this", exp.to_identifier(replacement))
            sql = self._to_sql(ast)
            return PerturbationResult(
                perturbed_sql=sql,
                error_type=self.error_type,
                description=f"Replaced column '{original_name}' (from {source_table}) with '{replacement}' (from {other_table})",
            )
        return None


class TableRedundancyOperator(PerturbationOperator):
    error_type = "table_redundancy"

    def inject(self, parsed_ast, db_tables, db_columns, db_path):
        ast = parsed_ast.copy()
        tables = list(ast.find_all(exp.Table))
        if not tables:
            return None

        real_tables = []
        try:
            conn = sqlite3.connect(db_path, timeout=1.0)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            real_tables = [row[0] for row in cursor.fetchall()]
            conn.close()
        except Exception:
            return None

        if len(real_tables) < 2:
            return None

        random.shuffle(tables)
        for table_node in tables:
            original_name = table_node.name
            if not original_name:
                continue
            others = [t for t in real_tables if _norm(t) != _norm(original_name)]
            if not others:
                continue
            replacement = random.choice(others)
            table_node.set("this", exp.to_identifier(replacement))
            sql = self._to_sql(ast)
            return PerturbationResult(
                perturbed_sql=sql,
                error_type=self.error_type,
                description=f"Replaced table '{original_name}' with unrelated schema table '{replacement}'",
            )
        return None


class TableMissingOperator(PerturbationOperator):
    error_type = "table_missing"

    def inject(self, parsed_ast, db_tables, db_columns, db_path):
        ast = parsed_ast.copy()
        joins = ast.args.get("joins")
        if not joins or not isinstance(joins, list) or len(joins) == 0:
            return None
        idx = random.randrange(len(joins))
        removed_join = joins.pop(idx)
        table_name = ""
        join_table = removed_join.find(exp.Table)
        if join_table:
            table_name = join_table.name
        sql = self._to_sql(ast)
        return PerturbationResult(
            perturbed_sql=sql,
            error_type=self.error_type,
            description=f"Removed JOIN on table '{table_name}'",
        )


class FunctionErrorOperator(PerturbationOperator):
    error_type = "function_error"

    def inject(self, parsed_ast, db_tables, db_columns, db_path):
        ast = parsed_ast.copy()

        func_nodes = list(ast.find_all(exp.Func))
        anon_nodes = list(ast.find_all(exp.Anonymous))

        candidates = func_nodes + anon_nodes
        if candidates:
            random.shuffle(candidates)
            for node in candidates:
                invalid_func = random.choice(INVALID_SQLITE_FUNCTIONS)
                if isinstance(node, exp.Anonymous):
                    original_name = node.name
                    node.set("this", invalid_func)
                else:
                    original_name = type(node).__name__
                    args = [node.this] if node.this else []
                    replacement = exp.Anonymous(this=invalid_func, expressions=args)
                    node.replace(replacement)
                sql = self._to_sql(ast)
                return PerturbationResult(
                    perturbed_sql=sql,
                    error_type=self.error_type,
                    description=f"Replaced function '{original_name}' with invalid '{invalid_func}'",
                )

        columns = list(ast.find_all(exp.Column))
        if columns:
            col_node = random.choice(columns)
            invalid_func = random.choice(INVALID_SQLITE_FUNCTIONS)
            original_name = col_node.name
            wrapper = exp.Anonymous(this=invalid_func, expressions=[col_node.copy()])
            col_node.replace(wrapper)
            sql = self._to_sql(ast)
            return PerturbationResult(
                perturbed_sql=sql,
                error_type=self.error_type,
                description=f"Wrapped column '{original_name}' in invalid function '{invalid_func}'",
            )
        return None


class ValueErrorOperator(PerturbationOperator):
    error_type = "value_error"

    def inject(self, parsed_ast, db_tables, db_columns, db_path):
        ast = parsed_ast.copy()
        where = ast.find(exp.Where)
        if where is None:
            return None

        literals = list(where.find_all(exp.Literal))
        if not literals:
            return None

        random.shuffle(literals)
        for lit in literals:
            if lit.is_string:
                original = lit.this
                lit.set("this", "WRONG_VALUE_12345")
                sql = self._to_sql(ast)
                return PerturbationResult(
                    perturbed_sql=sql,
                    error_type=self.error_type,
                    description=f"Replaced string literal '{original}' with 'WRONG_VALUE_12345'",
                )
            else:
                original = lit.this
                replacement = exp.Literal.string("not_a_number")
                lit.replace(replacement)
                sql = self._to_sql(ast)
                return PerturbationResult(
                    perturbed_sql=sql,
                    error_type=self.error_type,
                    description=f"Replaced numeric literal {original} with string 'not_a_number'",
                )
        return None


class ConditionErrorOperator(PerturbationOperator):
    error_type = "condition_error"

    def inject(self, parsed_ast, db_tables, db_columns, db_path):
        ast = parsed_ast.copy()
        where = ast.find(exp.Where)
        if where is None:
            return None

        for source_type, target_type in _OPERATOR_SWAPS.items():
            nodes = list(where.find_all(source_type))
            if nodes:
                node = random.choice(nodes)
                left = node.left
                right = node.right
                replacement = target_type(this=left.copy(), expression=right.copy())
                node.replace(replacement)
                sql = self._to_sql(ast)
                return PerturbationResult(
                    perturbed_sql=sql,
                    error_type=self.error_type,
                    description=f"Swapped operator {source_type.__name__} -> {target_type.__name__}",
                )
        return None


class ConditionMissingOperator(PerturbationOperator):
    error_type = "condition_missing"

    def inject(self, parsed_ast, db_tables, db_columns, db_path):
        ast = parsed_ast.copy()
        where = ast.find(exp.Where)
        if where is None:
            return None

        ast.set("where", None)
        sql = self._to_sql(ast)
        return PerturbationResult(
            perturbed_sql=sql,
            error_type=self.error_type,
            description="Removed WHERE clause entirely",
        )


class ClauseErrorOperator(PerturbationOperator):
    error_type = "clause_error"

    def inject(self, parsed_ast, db_tables, db_columns, db_path):
        ast = parsed_ast.copy()

        group = ast.args.get("group")
        if group is not None:
            ast.set("group", None)
            sql = self._to_sql(ast)
            return PerturbationResult(
                perturbed_sql=sql,
                error_type=self.error_type,
                description="Removed GROUP BY clause",
            )

        order = ast.args.get("order")
        if order is not None:
            ast.set("order", None)
            sql = self._to_sql(ast)
            return PerturbationResult(
                perturbed_sql=sql,
                error_type=self.error_type,
                description="Removed ORDER BY clause",
            )

        limit = ast.args.get("limit")
        if limit is not None:
            ast.set("limit", None)
            sql = self._to_sql(ast)
            return PerturbationResult(
                perturbed_sql=sql,
                error_type=self.error_type,
                description="Removed LIMIT clause",
            )

        return None


class ModifierErrorOperator(PerturbationOperator):
    error_type = "modifier_error"

    def inject(self, parsed_ast, db_tables, db_columns, db_path):
        ast = parsed_ast.copy()

        order = ast.find(exp.Order)
        if order is not None:
            ordered_exprs = list(order.find_all(exp.Ordered))
            if ordered_exprs:
                target = random.choice(ordered_exprs)
                is_desc = target.args.get("desc")
                if is_desc:
                    target.set("desc", False)
                    sql = self._to_sql(ast)
                    return PerturbationResult(
                        perturbed_sql=sql,
                        error_type=self.error_type,
                        description="Changed DESC to ASC in ORDER BY",
                    )
                else:
                    target.set("desc", True)
                    sql = self._to_sql(ast)
                    return PerturbationResult(
                        perturbed_sql=sql,
                        error_type=self.error_type,
                        description="Changed ASC to DESC in ORDER BY",
                    )

        select = ast.find(exp.Select)
        if select is None:
            return None

        distinct = select.args.get("distinct")
        if distinct:
            select.set("distinct", None)
            sql = self._to_sql(ast)
            return PerturbationResult(
                perturbed_sql=sql,
                error_type=self.error_type,
                description="Removed DISTINCT modifier",
            )
        else:
            select.set("distinct", exp.Distinct())
            sql = self._to_sql(ast)
            return PerturbationResult(
                perturbed_sql=sql,
                error_type=self.error_type,
                description="Added DISTINCT modifier",
            )


class TableMismatchOperator(PerturbationOperator):
    error_type = "table_mismatch"

    def inject(self, parsed_ast, db_tables, db_columns, db_path):
        ast = parsed_ast.copy()
        tables = list(ast.find_all(exp.Table))
        if not tables:
            return None

        real_tables = []
        try:
            conn = sqlite3.connect(db_path, timeout=1.0)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            real_tables = [row[0] for row in cursor.fetchall()]
            conn.close()
        except Exception:
            return None

        if len(real_tables) < 2:
            return None

        random.shuffle(tables)
        for table_node in tables:
            original_name = table_node.name
            if not original_name:
                continue
            others = [t for t in real_tables if _norm(t) != _norm(original_name)]
            if not others:
                continue
            replacement = random.choice(others)
            table_node.set("this", exp.to_identifier(replacement))
            sql = self._to_sql(ast)
            return PerturbationResult(
                perturbed_sql=sql,
                error_type=self.error_type,
                description=f"Replaced table '{original_name}' with different table '{replacement}'",
            )
        return None


ALL_OPERATORS: List[PerturbationOperator] = [
    AttributeRedundancyOperator(),
    AttributeMissingOperator(),
    AttributeMismatchOperator(),
    TableRedundancyOperator(),
    TableMissingOperator(),
    TableMismatchOperator(),
    FunctionErrorOperator(),
    ValueErrorOperator(),
    ConditionErrorOperator(),
    ConditionMissingOperator(),
    ClauseErrorOperator(),
    ModifierErrorOperator(),
]


def compose_errors(
    sql: str,
    operators: List[PerturbationOperator],
    db_tables: Set[str],
    db_columns: Set[str],
    db_path: str,
) -> Optional[PerturbationResult]:
    if not operators:
        return None
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None

    all_types = []
    all_descs = []
    current_sql = sql

    for op in operators:
        try:
            parsed = sqlglot.parse_one(current_sql, read="sqlite")
        except Exception:
            return None
        result = op.inject(parsed, db_tables, db_columns, db_path)
        if result is None:
            return None
        current_sql = result.perturbed_sql
        all_types.append(result.error_type)
        all_descs.append(result.description)

    return PerturbationResult(
        perturbed_sql=current_sql,
        error_type=",".join(all_types),
        description="Compound: " + " + ".join(f"[{d}]" for d in all_descs),
    )

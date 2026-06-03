import argparse
import json
import os
import re
from enum import Enum

import sqlglot
from sqlglot import exp
from tqdm import tqdm

_SPLIT_RE = re.compile(r"(dev|train)", flags=re.IGNORECASE)

_SKIP_ARG_KEYS = {
    "typed",
    "safe",
    "format",
    "action",
    "default",
    "catalog",
    "db",
    "hint",
    "kind",
    "operation_modifiers",
    "limit_options",
    "with_fill",
    "nested",
    "prefix",
}

_ARG_KEY_ORDER = {
    "this": 0,
    "left": 1,
    "right": 2,
    "expression": 3,
    "expressions": 4,
    "from_": 5,
    "joins": 6,
    "where": 7,
    "group": 8,
    "having": 9,
    "order": 10,
    "limit": 11,
    "offset": 12,
}


def _ordered_arg_items(node: exp.Expression):
    items = [(k, v) for k, v in node.args.items() if k not in _SKIP_ARG_KEYS]

    def sort_key(item):
        k = item[0]
        return (_ARG_KEY_ORDER.get(k, 10_000), k)

    items.sort(key=sort_key)
    return items


def _parse_number(text: str):
    try:
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        return float(text)
    except Exception:
        return text


def ast_to_compact_min(node, is_root: bool = False):
    if not isinstance(node, exp.Expression):
        return None

    if isinstance(node, exp.Identifier):
        return None

    if isinstance(node, exp.Table):
        payload = {"node_type": "table", "name": node.name, "children": []}
        if is_root:
            payload["sql_full"] = node.sql(dialect="sqlite")
        return payload

    if isinstance(node, exp.Column):
        payload = {"node_type": "column", "name": node.name, "children": []}
        if is_root:
            payload["sql_full"] = node.sql(dialect="sqlite")
        return payload

    if isinstance(node, exp.Null):
        payload = {"node_type": "literal", "value": None, "value_type": "null", "children": []}
        if is_root:
            payload["sql_full"] = node.sql(dialect="sqlite")
        return payload

    if isinstance(node, exp.Literal):
        if node.is_string:
            value_type = "string"
            value = node.this
        else:
            value_type = "number"
            value = _parse_number(str(node.this))
        payload = {"node_type": "literal", "value": value, "value_type": value_type, "children": []}
        if is_root:
            payload["sql_full"] = node.sql(dialect="sqlite")
        return payload

    if isinstance(node, exp.Paren):
        child = ast_to_compact_min(node.this, is_root=False) if node.this is not None else None
        children = [child] if child is not None else []
        payload = {"node_type": "paren", "children": children}
        if is_root:
            payload["sql_full"] = node.sql(dialect="sqlite")
        return payload

    if isinstance(node, exp.Cast):
        child = ast_to_compact_min(node.this, is_root=False) if node.this is not None else None
        children = [child] if child is not None else []
        to_expr = node.args.get("to")
        cast_to = None
        if isinstance(to_expr, exp.Expression):
            cast_to = to_expr.sql(dialect="sqlite")
        elif to_expr is not None:
            cast_to = str(to_expr)
        payload = {"node_type": "cast", "children": children}
        if cast_to is not None:
            payload["cast_to"] = cast_to
        if is_root:
            payload["sql_full"] = node.sql(dialect="sqlite")
        return payload

    children = []
    for _, arg_value in _ordered_arg_items(node):
        if arg_value is None:
            continue
        if isinstance(arg_value, list):
            if not arg_value:
                continue
            for item in arg_value:
                compact_child = ast_to_compact_min(item, is_root=False)
                if compact_child is not None:
                    children.append(compact_child)
        else:
            compact_child = ast_to_compact_min(arg_value, is_root=False)
            if compact_child is not None:
                children.append(compact_child)

    payload = {"node_type": node.key, "children": children}
    if is_root:
        payload["sql_full"] = node.sql(dialect="sqlite")
    return payload


def compact_to_sql(compact_node):
    if isinstance(compact_node, dict):
        return compact_node.get("sql_full", "")
    return ""

def ast_to_complete(node):
    if node is None:
        return None
    if isinstance(node, exp.Expression):
        payload = {"node_type": node.key}
        for k, v in node.args.items():
            out_k = "from_" if k == "from" else k
            payload[out_k] = ast_to_complete(v)
        return payload
    if isinstance(node, list):
        return [ast_to_complete(x) for x in node]
    
    if isinstance(node, Enum):
        return node.name
    if isinstance(node, (str, int, float, bool)):
        return node
    
    return str(node)


def _extract_split(input_path: str) -> str:
    base = os.path.basename(input_path)
    m = _SPLIT_RE.search(base)
    if not m:
        raise ValueError(f"Cannot infer split from filename: {base}")
    return m.group(1).lower()


def _normalize_sql_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    parts = text.split("\t----- bird -----\t", 1)
    return parts[0].strip()


def _parse_one_sql(sql: str, *, mode: str, dialect: str):
    if not sql:
        return None
    parsed = sqlglot.parse_one(sql, read=dialect)
    if mode == "complete":
        return ast_to_complete(parsed)
    return ast_to_compact_min(parsed, is_root=True)


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _build_out_path(*, input_path: str, output_dir: str, mode: str) -> str:
    split = _extract_split(input_path)
    suffix = "-complete" if mode == "complete" else ""
    return os.path.join(output_dir, f"ast_{split}{suffix}.json")


def _run(args) -> str:
    data = _load_json(args.input_path)
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = args.out_path or _build_out_path(
        input_path=args.input_path, output_dir=args.output_dir, mode=args.mode
    )

    n = args.n
    if isinstance(data, dict):
        items = list(data.items())
        all_numeric_keys = True
        for k, _ in items:
            if not isinstance(k, str) or not k.isdigit():
                all_numeric_keys = False
                break
        if all_numeric_keys:
            items.sort(key=lambda kv: int(kv[0]))
        if n >= 0:
            items = items[:n]
        out_list = []
        for k, v in tqdm(items, desc="Building AST"):
            sql = _normalize_sql_text(v)
            try:
                out_list.append(_parse_one_sql(sql, mode=args.mode, dialect=args.dialect))
            except Exception:
                out_list.append(None)
        with open(out_path, "w") as f:
            json.dump(out_list, f, ensure_ascii=False, indent=2)
        return out_path

    if isinstance(data, list):
        out_list = []
        entries = data if n < 0 else data[:n]
        for entry in tqdm(entries, desc="Building AST"):
            sql = entry.get("SQL") if isinstance(entry, dict) else None
            sql = _normalize_sql_text(sql) if sql else ""
            try:
                out_list.append(_parse_one_sql(sql, mode=args.mode, dialect=args.dialect))
            except Exception:
                out_list.append(None)
        with open(out_path, "w") as f:
            json.dump(out_list, f, ensure_ascii=False, indent=2)
        return out_path

    raise TypeError(f"Unsupported JSON root type: {type(data)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--mode", choices=["default", "complete"], default="default")
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "finetune", "ast", "bird"),
    )
    parser.add_argument("--out_path", default=None)
    parser.add_argument("--dialect", default="sqlite")
    parser.add_argument("--n", type=int, default=-1)
    args = parser.parse_args()

    out_path = _run(args)
    print(out_path)


if __name__ == "__main__":
    main()

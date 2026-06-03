import json
import os
import sqlite3
from typing import Dict, List, Optional, Set, Tuple


def load_gold_data(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def load_predicted_sqls(path: str) -> Dict[int, Tuple[str, str]]:
    with open(path, "r") as f:
        raw = json.load(f)
    result = {}
    for idx_str, value in raw.items():
        if isinstance(value, str) and "\t----- bird -----\t" in value:
            sql, db_id = value.split("\t----- bird -----\t", 1)
            result[int(idx_str)] = (sql.strip(), db_id.strip())
        else:
            result[int(idx_str)] = (str(value).strip(), "")
    return result


def get_db_path(db_root: str, db_id: str) -> str:
    return os.path.join(db_root, db_id, f"{db_id}.sqlite")


def execute_sql(
    sql: str, db_path: str, timeout: float = 30.0
) -> Tuple[Optional[List], Optional[str]]:
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results, None
    except Exception as e:
        return None, str(e)


def get_db_schema(db_path: str) -> Tuple[Set[str], Set[str]]:
    tables: Set[str] = set()
    columns: Set[str] = set()
    try:
        conn = sqlite3.connect(db_path, timeout=1.0)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for row in cursor.fetchall():
            table_name = row[0]
            tables.add(table_name.lower().replace(" ", "_").replace("-", "_"))
            try:
                cursor.execute(f'PRAGMA table_info("{table_name}")')
                for col_row in cursor.fetchall():
                    columns.add(
                        col_row[1].lower().replace(" ", "_").replace("-", "_")
                    )
            except Exception:
                pass
        conn.close()
    except Exception:
        pass
    return tables, columns


def get_db_schema_detailed(db_path: str) -> Dict[str, List[str]]:
    schema: Dict[str, List[str]] = {}
    try:
        conn = sqlite3.connect(db_path, timeout=1.0)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for row in cursor.fetchall():
            table_name = row[0]
            cols = []
            try:
                cursor.execute(f'PRAGMA table_info("{table_name}")')
                for col_row in cursor.fetchall():
                    cols.append(col_row[1])
            except Exception:
                pass
            schema[table_name] = cols
        conn.close()
    except Exception:
        pass
    return schema


def get_schema_ddl(db_path: str) -> str:
    ddl_parts = []
    try:
        conn = sqlite3.connect(db_path, timeout=1.0)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        )
        for row in cursor.fetchall():
            ddl_parts.append(row[0])
        conn.close()
    except Exception:
        pass
    return "\n\n".join(ddl_parts)


def validate_injection(
    original_sql: str, perturbed_sql: str, db_path: str
) -> Tuple[bool, str]:
    orig_res, orig_err = execute_sql(original_sql, db_path)
    pert_res, pert_err = execute_sql(perturbed_sql, db_path)

    if orig_err is not None:
        return False, "invalid"

    if pert_err is not None:
        return True, "hard_error"

    if set(orig_res) != set(pert_res):
        return True, "soft_error"

    return False, "invalid"

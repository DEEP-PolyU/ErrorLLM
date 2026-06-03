import os
import csv
import json
import sqlite3
import functools
from typing import Dict, List, Optional, Tuple, Any


MAX_EXAMPLE_VALUES = 3


def load_tables(tables_path: str) -> Dict[str, dict]:
    
    tables_list = json.load(open(tables_path, "r"))
    return {t["db_id"]: t for t in tables_list}


def load_column_descriptions(db_desc_dir: str, table_names_original: List[str]) -> Dict[str, Dict[str, str]]:
    descs: Dict[str, Dict[str, str]] = {}
    if not os.path.isdir(db_desc_dir):
        return descs
    for tname in table_names_original:
        csv_path = os.path.join(db_desc_dir, f"{tname}.csv")
        if not os.path.isfile(csv_path):
            continue
        tname_lower = tname.lower()
        descs[tname_lower] = {}
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    orig_col = row.get("original_column_name", "").strip()
                    desc = row.get("column_description", "").strip()
                    if orig_col:
                        descs[tname_lower][orig_col.lower()] = desc
        except Exception:
            pass
    return descs


@functools.lru_cache(maxsize=5000, typed=False)
def fetch_column_examples(
    db_path: str,
    table_name_original: str,
    column_name_original: str,
    limit: int = MAX_EXAMPLE_VALUES,
) -> Tuple:

    if not os.path.isfile(db_path):
        return ()
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = f'SELECT DISTINCT "{column_name_original}" FROM "{table_name_original}" WHERE "{column_name_original}" IS NOT NULL LIMIT {limit}'
        cursor.execute(query)
        values = tuple(row[0] for row in cursor.fetchall())
        conn.close()
        return values
    except Exception:
        return ()


def build_column_id(table_name_original: str, column_name_original: str) -> str:

    import re as _re
    t = table_name_original.lower().replace(" ", "_")
    c = column_name_original.lower().replace(" ", "_")
    
    c = c.replace("/", "_")
    
    c = _re.sub(r'(\d)-(\d)', r'\1_\2', c)
    
    c = _re.sub(r'([a-z])-([a-z])', r'\1_\2', c)
    
    c = c.replace("-", "")
    
    for ch in "()#%":
        c = c.replace(ch, "")
    
    while "__" in c:
        c = c.replace("__", "_")
    c = c.strip("_")
    return f"{t}.{c}"


def build_database_structure(
    db_id: str,
    table_meta: dict,
    db_dir: str,
    column_descs: Optional[Dict[str, Dict[str, str]]] = None,
    include_descriptions: bool = True,
    include_examples: bool = True,
    linked_tables_columns: Optional[Dict[str, Optional[List[str]]]] = None,
) -> Tuple[List[dict], List[dict]]:

    table_names_original = table_meta["table_names_original"]
    column_names_original = table_meta["column_names_original"]
    column_names = table_meta["column_names"]
    column_types = table_meta["column_types"]
    primary_keys = table_meta.get("primary_keys", [])
    foreign_keys = table_meta.get("foreign_keys", [])

    sqlite_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")

    if column_descs is None and include_descriptions:
        db_desc_dir = os.path.join(db_dir, db_id, "database_description")
        column_descs = load_column_descriptions(db_desc_dir, table_names_original)

    col_index = []
    for cidx, (tid_name, ctype) in enumerate(zip(column_names_original, column_types)):
        tid, cname_orig = tid_name
        _, clabel = column_names[cidx]
        col_index.append((tid, cname_orig, clabel, ctype))

    if linked_tables_columns is not None:
        included_tables = set()
        included_columns = set()
        for tname, cols in linked_tables_columns.items():
            if cols is None:
                continue
            tname_lower = tname.lower()
            
            tidx = None
            for i, t in enumerate(table_names_original):
                if t.lower() == tname_lower:
                    tidx = i
                    break
            if tidx is None:
                continue
            included_tables.add(tidx)
            for col_lower in cols:
                
                for cidx, (tid, cname_orig, clabel, ctype) in enumerate(col_index):
                    if cidx == 0:
                        continue  
                    if tid == tidx and (cname_orig.lower() == col_lower or clabel.lower() == col_lower):
                        included_columns.add(cidx)

        pk_set = set()
        for pk in primary_keys:
            if isinstance(pk, list):
                pk_set.update(pk)
            else:
                pk_set.add(pk)
        for cidx in pk_set:
            if cidx < len(col_index):
                tid = col_index[cidx][0]
                if tid in included_tables:
                    included_columns.add(cidx)

        for fk_from, fk_to in foreign_keys:
            if fk_from < len(col_index) and fk_to < len(col_index):
                tid_from = col_index[fk_from][0]
                tid_to = col_index[fk_to][0]
                if tid_from in included_tables and tid_to in included_tables:
                    included_columns.add(fk_from)
                    included_columns.add(fk_to)
    else:
        
        included_tables = set(range(len(table_names_original)))
        included_columns = set(range(1, len(column_names_original)))

    nodes = []
    edges = []
    table_id_map = {} 
    column_id_map = {} 

    for tidx in sorted(included_tables):
        tname_orig = table_names_original[tidx]
        node_id = tname_orig.lower()
        node = {
            "id": node_id,
            "type": "table",
            "label": tname_orig.lower(),
        }
        nodes.append(node)
        table_id_map[tidx] = node_id

    
    for cidx in sorted(included_columns):
        tid, cname_orig, clabel, ctype = col_index[cidx]
        if tid not in table_id_map:
            
            tname_orig = table_names_original[tid]
            node_id = tname_orig.lower()
            table_id_map[tid] = node_id
            nodes.insert(len([n for n in nodes if n["type"] == "table"]), {
                "id": node_id,
                "type": "table",
                "label": tname_orig.lower(),
            })
            included_tables.add(tid)

        tname_orig = table_names_original[tid]
        col_node_id = build_column_id(tname_orig, cname_orig)
        col_node = {
            "id": col_node_id,
            "type": "column",
            "label": cname_orig,
            "datatype": ctype,
        }
        
        if include_descriptions and column_descs:
            tname_lower = tname_orig.lower()
            cname_lower = cname_orig.lower()
            desc = ""
            if tname_lower in column_descs:
                desc = column_descs[tname_lower].get(cname_lower, "")
            if desc:
                col_node["description"] = desc

        if include_examples:
            examples = fetch_column_examples(sqlite_path, tname_orig, cname_orig)
            if examples:
                col_node["examples"] = list(examples)

        nodes.append(col_node)
        column_id_map[cidx] = col_node_id

    for cidx in sorted(included_columns):
        tid = col_index[cidx][0]
        if tid in table_id_map and cidx in column_id_map:
            edges.append({
                "src": table_id_map[tid],
                "tgt": column_id_map[cidx],
                "rel": "has_column",
            })

    for tidx in sorted(included_tables):
        edges.append({
            "src": db_id,
            "tgt": table_id_map[tidx],
            "rel": "has_table",
        })

    pk_set = set()
    for pk in primary_keys:
        if isinstance(pk, list):
            pk_set.update(pk)
        else:
            pk_set.add(pk)
    for cidx in sorted(included_columns):
        if cidx in pk_set and cidx in column_id_map:
            tid = col_index[cidx][0]
            if tid in table_id_map:
                edges.append({
                    "src": column_id_map[cidx],
                    "tgt": table_id_map[tid],
                    "rel": "primary_key_of",
                })

    for fk_from, fk_to in foreign_keys:
        if fk_from in column_id_map and fk_to in column_id_map:
            edges.append({
                "src": column_id_map[fk_from],
                "tgt": column_id_map[fk_to],
                "rel": "foreign_key_to",
            })

    return nodes, edges

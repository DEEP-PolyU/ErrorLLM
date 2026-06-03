import os
import sys
import json
import time
import argparse
from typing import Dict, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schema_utils import (
    load_tables,
    load_column_descriptions,
    build_database_structure,
)


def _resolve_target_indices(
    linking_result: Dict[str, Optional[List[str]]],
    table_meta: dict,
) -> tuple:
    table_names_original = table_meta["table_names_original"]
    column_names_original = table_meta["column_names_original"]
    column_names = table_meta["column_names"]

    target_tables: Set[int] = set()
    target_columns: Set[int] = set()

    for tname, cols in linking_result.items():
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
        target_tables.add(tidx)
        for col_lower in cols:
            for cidx, (tid, cname_orig) in enumerate(column_names_original):
                if cidx == 0:
                    continue
                _, clabel = column_names[cidx]
                if tid == tidx and (cname_orig.lower() == col_lower
                                    or clabel.lower() == col_lower):
                    target_columns.add(cidx)
                    break

    return target_tables, target_columns


def build_qss_entry(
    entry: dict,
    table_meta: dict,
    db_dir: str,
    mode: str,
    schema_linker,
    linking_result: Optional[Dict[str, Optional[List[str]]]] = None,
    include_descriptions: bool = True,
    include_examples: bool = True,
    value_match: bool = True,
) -> dict:
    db_id = entry["db_id"]
    question = entry["question"]
    sqlite_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
    db_path = sqlite_path if os.path.isfile(sqlite_path) and value_match else None

    linked_tc = None
    if mode == "schema_linking" and linking_result is not None:
        linked_tc = linking_result

    db_desc_dir = os.path.join(db_dir, db_id, "database_description")
    column_descs = load_column_descriptions(
        db_desc_dir, table_meta["table_names_original"])

    schema_nodes, schema_edges = build_database_structure(
        db_id=db_id,
        table_meta=table_meta,
        db_dir=db_dir,
        column_descs=column_descs,
        include_descriptions=include_descriptions,
        include_examples=include_examples,
        linked_tables_columns=linked_tc,
    )

    target_table_idxs = None
    target_column_idxs = None
    if mode == "schema_linking" and linking_result is not None:
        target_table_idxs, target_column_idxs = _resolve_target_indices(
            linking_result, table_meta)

    q_nodes, q_edges = schema_linker.build_linking(
        question=question,
        table_meta=table_meta,
        db_path=db_path,
        target_table_idxs=target_table_idxs,
        target_column_idxs=target_column_idxs,
    )

    schema_node_ids = {n["id"] for n in schema_nodes}
    valid_q_edges = []
    for e in q_edges:
        if e["tgt"] in schema_node_ids:
            valid_q_edges.append(e)

    all_nodes = schema_nodes + q_nodes
    all_edges = schema_edges + valid_q_edges

    qss_entry = {
        "question_id": entry.get("question_id", 0),
        "db_id": db_id,
        "question": question,
        "question_schema_structure": {
            "db": db_id,
            "nodes": all_nodes,
            "edges": all_edges,
        },
    }
    return qss_entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["scratch", "schema_linking"]
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--tables_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--linking_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--plm_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--no_descriptions",
        action="store_true",
    )
    parser.add_argument(
        "--no_examples",
        action="store_true",
    )
    parser.add_argument(
        "--no_value_match",
        action="store_true",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    if args.mode == "schema_linking" and args.linking_path is None:
        parser.error("--linking_path is required for schema_linking mode.")

    if args.plm_path is None:
        parser.error("--plm_path is required.")

    print(f"Mode: {args.mode}")
    print(f"Loading dataset from {args.dataset_path} ...")
    dataset = json.load(open(args.dataset_path, "r"))
    if args.limit:
        dataset = dataset[:args.limit]
        print(f"  Limited to first {args.limit} entries.")
    print(f"  {len(dataset)} entries loaded.")

    print(f"Loading tables from {args.tables_path} ...")
    tables = load_tables(args.tables_path)
    print(f"  {len(tables)} databases loaded.")

    linking_data = None
    if args.linking_path:
        print(f"Loading schema linking from {args.linking_path} ...")
        linking_data = json.load(open(args.linking_path, "r"))
        if args.limit:
            linking_data = linking_data[:args.limit]
        print(f"  {len(linking_data)} linking entries loaded.")
        if len(linking_data) != len(dataset):
            print(f"  WARNING: linking entries ({len(linking_data)}) != "
                  f"dataset entries ({len(dataset)})")

    print(f"Initializing RGAT encoder from PLM: {args.plm_path} ...")
    print(f"  hidden_size={args.hidden_size}, num_heads={args.num_heads}, "
          f"num_layers={args.num_layers}")
    from transformers import AutoTokenizer
    from rgat_encoder import RGATEncoder
    from linking_utils import SchemaLinker

    tokenizer = AutoTokenizer.from_pretrained(
        args.plm_path, add_prefix_space=False)
    encoder = RGATEncoder(
        plm_path=args.plm_path,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.2,
    )
    print("  RGAT encoder initialized.")

    device = 'cuda' if args.use_gpu else 'cpu'
    encoder.to(device)
    print(f"  Device: {device}")

    print(f"Initializing SchemaLinker (threshold={args.similarity_threshold})...")
    schema_linker = SchemaLinker(
        encoder=encoder,
        tokenizer=tokenizer,
        db_dir=args.db_dir,
        use_gpu=args.use_gpu,
        similarity_threshold=args.similarity_threshold,
    )
    print("  SchemaLinker ready.")

    print("Building QSS with PLM-based RGAT schema linking ...")
    results = []
    t0 = time.time()
    for idx, entry in enumerate(dataset):
        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - t0
            print(f"  Processing {idx + 1}/{len(dataset)} ... "
                  f"({elapsed:.1f}s elapsed)")

        db_id = entry["db_id"]
        if db_id not in tables:
            print(f"  WARNING: db_id '{db_id}' not found in tables, skipping.")
            continue

        table_meta = tables[db_id]
        linking_result = linking_data[idx] if linking_data else None

        qss = build_qss_entry(
            entry=entry,
            table_meta=table_meta,
            db_dir=args.db_dir,
            mode=args.mode,
            schema_linker=schema_linker,
            linking_result=linking_result,
            include_descriptions=not args.no_descriptions,
            include_examples=not args.no_examples,
            value_match=not args.no_value_match,
        )
        results.append(qss)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    elapsed = time.time() - t0
    print(f"Done. {len(results)} QSS entries written to {args.output_path} "
          f"({elapsed:.1f}s total)")

    total_q_nodes = sum(
        len([n for n in r["question_schema_structure"]["nodes"]
             if n.get("type") == "question_entity"])
        for r in results
    )
    total_q_edges = sum(
        len([e for e in r["question_schema_structure"]["edges"]
             if e.get("rel", "").startswith("relates_to")])
        for r in results
    )
    print(f"Summary: {total_q_nodes} question entity nodes, "
          f"{total_q_edges} linking edges across {len(results)} entries.")


if __name__ == "__main__":
    main()

import argparse
import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import sqlglot

from .utils import (
    execute_sql,
    get_db_path,
    get_db_schema,
    get_schema_ddl,
    load_gold_data,
    load_predicted_sqls,
    validate_injection,
)
from .perturbation_operators import (
    ALL_OPERATORS,
    PerturbationOperator,
    PerturbationResult,
    compose_errors,
)

ERROR_COUNT_WEIGHTS = {1: 4, 2: 5, 3: 1.5}

SINGLE_TYPE_WEIGHTS = {
    
    "attribute_mismatch": 3,
    "value_error":        3,
    "condition_error":    3,
    
    "function_error":     2,
    "condition_missing":  2,
    "clause_error":       2,
    "table_redundancy":   2,
    "table_mismatch":     2,
    
    "modifier_error":     1,
    "table_missing":      1,
    "attribute_missing":  1,
    "attribute_redundancy": 1,
}


def _build_operator_index(operators):
    
    return {op.error_type: op for op in operators}


def _weighted_choice(weights_dict):
    
    keys = list(weights_dict.keys())
    vals = [weights_dict[k] for k in keys]
    return random.choices(keys, weights=vals, k=1)[0]


def _sample_n_operators(operator_index, n):
    
    available = {
        t: w for t, w in SINGLE_TYPE_WEIGHTS.items() if t in operator_index
    }
    if len(available) < n:
        return None
    chosen = []
    pool = dict(available)
    for _ in range(n):
        t = _weighted_choice(pool)
        chosen.append(operator_index[t])
        del pool[t]          
    return chosen

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _build_sample(
    gold_item: Dict,
    original_sql: str,
    result: PerturbationResult,
    status: str,
    method: str,
) -> Dict:
    error_types = result.error_type.split(",") if "," in result.error_type else [result.error_type]
    return {
        "question_id": gold_item["question_id"],
        "db_id": gold_item["db_id"],
        "question": gold_item.get("question", ""),
        "evidence": gold_item.get("evidence", ""),
        "original_sql": original_sql,
        "erroneous_sql": result.perturbed_sql,
        "error_types": error_types,
        "synthesis_method": method,
        "validation_status": status,
        "description": result.description,
    }


def _build_llm_sample(
    gold_item: Dict,
    predicted_sql: str,
    annotation: Dict,
) -> Dict:
    return {
        "question_id": gold_item["question_id"],
        "db_id": gold_item["db_id"],
        "question": gold_item.get("question", ""),
        "evidence": gold_item.get("evidence", ""),
        "original_sql": gold_item["SQL"],
        "erroneous_sql": predicted_sql,
        "error_types": annotation["error_types"],
        "synthesis_method": "llm_assisted",
        "validation_status": "verified",
        "refined_sql": annotation.get("refined_sql", ""),
        "llm_explanation": annotation.get("explanation", ""),
    }


class DataSynthesizer:
    def __init__(
        self,
        db_root: str,
        gold_data: List[Dict],
        predicted_sqls: Dict[int, Tuple[str, str]],
        operators: Optional[List[PerturbationOperator]] = None,
        skip_llm: bool = False,
    ):
        self.db_root = db_root
        self.gold_data = gold_data
        self.predicted_sqls = predicted_sqls
        self.operators = operators or ALL_OPERATORS
        self.skip_llm = skip_llm
        self._schema_cache: Dict[str, Tuple] = {}

    def _get_schema(self, db_id: str):
        if db_id not in self._schema_cache:
            db_path = get_db_path(self.db_root, db_id)
            self._schema_cache[db_id] = get_db_schema(db_path)
        return self._schema_cache[db_id]

    def run(
        self,
        output_path: str,
        num_samples: int = -1,
        skip_llm: bool = False,
    ):
        results: List[Dict] = []
        items = self.gold_data if num_samples < 0 else self.gold_data[:num_samples]
        op_index = _build_operator_index(self.operators)

        logger.info("Part A: Rule-based perturbation from gold SQLs (%d items)", len(items))
        for i, item in enumerate(items):
            gold_sql = item["SQL"]
            db_id = item["db_id"]
            db_path = get_db_path(self.db_root, db_id)
            db_tables, db_columns = self._get_schema(db_id)

            try:
                parsed = sqlglot.parse_one(gold_sql, read="sqlite")
            except Exception as e:
                logger.warning("Failed to parse gold SQL #%d: %s", i, e)
                continue

            n_errors = _weighted_choice(ERROR_COUNT_WEIGHTS)

            if n_errors == 1:
                chosen_type = _weighted_choice(SINGLE_TYPE_WEIGHTS)
                op = op_index.get(chosen_type)
                if op is None:
                    continue
                try:
                    result = op.inject(parsed.copy(), db_tables, db_columns, db_path)
                except Exception as e:
                    logger.debug("Operator %s failed on #%d: %s", chosen_type, i, e)
                    result = None
                if result is not None:
                    valid, status = validate_injection(gold_sql, result.perturbed_sql, db_path)
                    if valid:
                        results.append(_build_sample(item, gold_sql, result, status, "rule_based"))

            else:
                for _attempt in range(3):
                    ops = _sample_n_operators(op_index, n_errors)
                    if ops is None:
                        break
                    try:
                        compound = compose_errors(
                            gold_sql, ops, db_tables, db_columns, db_path
                        )
                    except Exception:
                        compound = None
                    if compound:
                        valid, status = validate_injection(gold_sql, compound.perturbed_sql, db_path)
                        if valid:
                            results.append(
                                _build_sample(item, gold_sql, compound, status, "rule_based_compound")
                            )
                            break

            if (i + 1) % 100 == 0:
                logger.info("  Part A progress: %d/%d", i + 1, len(items))

        logger.info("Part A done: %d samples generated", len(results))

        pred_items = self.predicted_sqls if num_samples < 0 else {
            k: v for k, v in self.predicted_sqls.items() if k < num_samples
        }
        logger.info("Part B: Rule-based from correct predictions (%d items)", len(pred_items))
        part_b_count = 0
        for idx, (pred_sql, db_id) in pred_items.items():
            if idx >= len(self.gold_data):
                continue
            gold_item = self.gold_data[idx]
            db_path = get_db_path(self.db_root, db_id)
            db_tables, db_columns = self._get_schema(db_id)

            pred_res, pred_err = execute_sql(pred_sql, db_path)
            gold_res, gold_err = execute_sql(gold_item["SQL"], db_path)
            if pred_err or gold_err:
                continue
            if pred_res is None or gold_res is None:
                continue
            if set(pred_res) != set(gold_res):
                continue  

            try:
                parsed = sqlglot.parse_one(pred_sql, read="sqlite")
            except Exception:
                continue

            for op in self.operators:
                try:
                    result = op.inject(parsed.copy(), db_tables, db_columns, db_path)
                except Exception:
                    continue
                if result is None:
                    continue
                valid, status = validate_injection(pred_sql, result.perturbed_sql, db_path)
                if valid:
                    results.append(
                        _build_sample(gold_item, pred_sql, result, status, "rule_based_from_pred")
                    )
                    part_b_count += 1

        logger.info("Part B done: %d additional samples", part_b_count)

        if not skip_llm and not self.skip_llm:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set, skipping Part C")
            else:
                logger.info("Part C: LLM-assisted error annotation")
                try:
                    from .llm_error_injection import ErrorLLMAnnotator
                    annotator = ErrorLLMAnnotator(api_key=api_key)
                except Exception as e:
                    logger.error("Failed to initialize LLM annotator: %s", e)
                    annotator = None

                if annotator:
                    part_c_count = 0
                    part_c_tried = 0
                    for idx, (pred_sql, db_id) in pred_items.items():
                        if idx >= len(self.gold_data):
                            continue
                        gold_item = self.gold_data[idx]
                        db_path = get_db_path(self.db_root, db_id)

                        pred_res, pred_err = execute_sql(pred_sql, db_path)
                        gold_res, gold_err = execute_sql(gold_item["SQL"], db_path)
                        is_correct = (
                            pred_err is None
                            and gold_err is None
                            and pred_res is not None
                            and gold_res is not None
                            and set(pred_res) == set(gold_res)
                        )
                        if is_correct:
                            continue

                        schema_str = get_schema_ddl(db_path)
                        part_c_tried += 1
                        try:
                            annotation = annotator.annotate(
                                question=gold_item.get("question", ""),
                                schema_str=schema_str,
                                predicted_sql=pred_sql,
                                ground_truth_sql=gold_item["SQL"],
                                db_path=db_path,
                                evidence=gold_item.get("evidence", ""),
                            )
                        except Exception as e:
                            logger.debug("LLM annotation failed for #%d: %s", idx, e)
                            annotation = None

                        if annotation:
                            results.append(_build_llm_sample(gold_item, pred_sql, annotation))
                            part_c_count += 1

                        if part_c_tried % 50 == 0:
                            logger.info("  Part C progress: tried %d, accepted %d", part_c_tried, part_c_count)

                    logger.info("Part C done: %d/%d accepted", part_c_count, part_c_tried)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d synthesized samples to %s", len(results), output_path)

        method_counts: Dict[str, int] = {}
        error_type_counts: Dict[str, int] = {}
        status_counts: Dict[str, int] = {}
        for r in results:
            m = r.get("synthesis_method", "unknown")
            method_counts[m] = method_counts.get(m, 0) + 1
            s = r.get("validation_status", "unknown")
            status_counts[s] = status_counts.get(s, 0) + 1
            for et in r.get("error_types", []):
                error_type_counts[et] = error_type_counts.get(et, 0) + 1

        print(f"\nTotal synthesized samples: {len(results)}")
        print("\nBy method:")
        for m, c in sorted(method_counts.items()):
            print(f"  {m}: {c}")
        print("\nBy validation status:")
        for s, c in sorted(status_counts.items()):
            print(f"  {s}: {c}")
        print("\nBy error type:")
        for et, c in sorted(error_type_counts.items(), key=lambda x: -x[1]):
            print(f"  {et}: {c}")


def main():
    parser = argparse.ArgumentParser(description="ErrorLLM Training Data Synthesis")
    parser.add_argument("--db_root", required=True)
    parser.add_argument("--gold_path", required=True)
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Limit number of gold samples to process (-1 for all)")
    parser.add_argument("--skip_llm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    logger.info("Loading gold data from %s", args.gold_path)
    gold_data = load_gold_data(args.gold_path)
    logger.info("Loaded %d gold items", len(gold_data))

    logger.info("Loading predicted SQLs from %s", args.pred_path)
    predicted_sqls = load_predicted_sqls(args.pred_path)
    logger.info("Loaded %d predicted items", len(predicted_sqls))

    synthesizer = DataSynthesizer(
        db_root=args.db_root,
        gold_data=gold_data,
        predicted_sqls=predicted_sqls,
        skip_llm=args.skip_llm,
    )

    synthesizer.run(
        output_path=args.output,
        num_samples=args.num_samples,
        skip_llm=args.skip_llm,
    )


if __name__ == "__main__":
    main()

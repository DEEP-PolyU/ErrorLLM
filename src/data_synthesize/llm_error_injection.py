import json
import logging
import time
from typing import Dict, List, Optional, Tuple

from .utils import execute_sql

logger = logging.getLogger(__name__)

VALID_ERROR_TYPES = {
    "attribute_mismatch",
    "attribute_redundancy",
    "attribute_missing",
    "table_mismatch",
    "table_redundancy",
    "table_missing",
    "value_error",
    "condition_missing",
    "condition_error",
    "function_error",
    "clause_error",
    "modifier_error",
}

ERROR_TAXONOMY = """Error Type Taxonomy (12 types):
1. attribute_mismatch - Wrong column used (column exists in schema but belongs to wrong table or context)
2. attribute_redundancy - Column referenced does not exist in the database schema at all
3. attribute_missing - A required column is missing from the query (e.g., missing from SELECT or WHERE)
4. table_mismatch - Wrong table selected (table exists but is incorrect for this query)
5. table_redundancy - Table referenced does not exist in the database schema at all
6. table_missing - A required table is missing from FROM/JOIN clauses
7. value_error - Wrong literal value or data type mismatch in conditions
8. condition_missing - A required WHERE or HAVING condition is absent
9. condition_error - Wrong comparison operator or logical structure in conditions
10. function_error - Wrong, invalid, or misused SQL function (e.g., using MySQL function in SQLite)
11. clause_error - Missing or incorrect GROUP BY, ORDER BY, LIMIT, or HAVING clause
12. modifier_error - Missing or incorrect DISTINCT, ASC/DESC modifier"""


SYSTEM_PROMPT = f"""You are an expert SQL error analyst. Given a natural language question, database schema, a predicted SQL query (which may contain errors), and the ground-truth SQL query, your task is to:

1. Identify the specific error types present in the predicted SQL compared to the ground truth.
2. Produce a refined SQL that fixes the errors in the predicted SQL while maintaining its overall structure.

{ERROR_TAXONOMY}

IMPORTANT:
- The refined SQL MUST produce the same execution results as the ground truth SQL (refine to be correct).
- Return your answer as a JSON object with exactly these fields:
  - "error_types": array of error type strings from the taxonomy above
  - "refined_sql": the corrected SQL query string
  - "explanation": brief explanation of what errors were found and how they were fixed"""


class ErrorLLMAnnotator:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", max_retries: int = 3):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def annotate(
        self,
        question: str,
        schema_str: str,
        predicted_sql: str,
        ground_truth_sql: str,
        db_path: str,
        evidence: str = "",
    ) -> Optional[Dict]: 
        messages = self._build_prompt(
            question, schema_str, predicted_sql, ground_truth_sql, evidence
        )

        response = self._call_llm(messages)
        if response is None:
            return None

        parsed = self._parse_response(response)
        if parsed is None:
            return None

        refined_sql, error_types, explanation = parsed

        if not self._validate(refined_sql, ground_truth_sql, db_path):
            logger.debug("Execution verification failed for refined SQL")
            return None

        return {
            "error_types": error_types,
            "refined_sql": refined_sql,
            "explanation": explanation,
        }

    def _build_prompt(
        self,
        question: str,
        schema_str: str,
        predicted_sql: str,
        ground_truth_sql: str,
        evidence: str,
    ) -> List[Dict]:
        user_content = f"""Database Schema:
{schema_str}

Question: {question}
"""
        if evidence:
            user_content += f"Evidence: {evidence}\n"

        user_content += f"""
Predicted SQL (may contain errors):
{predicted_sql}

Ground Truth SQL:
{ground_truth_sql}

Analyze the errors in the predicted SQL compared to the ground truth. Return a JSON object with "error_types" (array of error type strings), "refined_sql" (corrected SQL), and "explanation" (brief description of errors found)."""

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _call_llm(self, messages: List[Dict]) -> Optional[str]:
        client = self._get_client()
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content
            except Exception as e:
                wait = 2 ** attempt
                logger.warning("LLM call failed (attempt %d/%d): %s. Retrying in %ds",
                               attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)
        return None

    def _parse_response(self, response: str) -> Optional[Tuple[str, List[str], str]]:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            logger.debug("Failed to parse LLM response as JSON")
            return None

        refined_sql = data.get("refined_sql", "").strip()
        error_types_raw = data.get("error_types", [])
        explanation = data.get("explanation", "")

        if not refined_sql:
            logger.debug("No refined_sql in LLM response")
            return None

        if not isinstance(error_types_raw, list) or not error_types_raw:
            logger.debug("No error_types in LLM response")
            return None

        error_types = []
        for et in error_types_raw:
            et_norm = str(et).strip().lower().replace(" ", "_").replace("-", "_")
            if et_norm in VALID_ERROR_TYPES:
                error_types.append(et_norm)
            else:
                logger.debug("Unknown error type from LLM: %s", et)

        if not error_types:
            logger.debug("No valid error types after filtering")
            return None

        return refined_sql, error_types, explanation

    def _validate(self, refined_sql: str, ground_truth_sql: str, db_path: str) -> bool:
        refined_res, refined_err = execute_sql(refined_sql, db_path)
        gt_res, gt_err = execute_sql(ground_truth_sql, db_path)

        if refined_err is not None or gt_err is not None:
            return False
        if refined_res is None or gt_res is None:
            return False
        return set(refined_res) == set(gt_res)

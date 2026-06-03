import re
from typing import Optional

from openai import OpenAI

from .types import OrderedRefinementContext, RefinementResult


class RefLLM:

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None, temperature: Optional[float] = None):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def refine(self, context: OrderedRefinementContext) -> RefinementResult:
        if not context.error_contexts:
            return RefinementResult(
                original_sql=context.full_sql,
                refined_sql=context.full_sql,
                success=True
            )

        prompt = self._build_prompt(context)

        try:
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert SQL Refiner. Fix all errors in the SQL query precisely."},
                    {"role": "user", "content": prompt}
                ],
            }

            if self.temperature is not None:
                kwargs["temperature"] = self.temperature

            response = self.client.chat.completions.create(**kwargs)

            result_text = response.choices[0].message.content
            refined_sql = self._extract_sql(result_text)

            return RefinementResult(
                original_sql=context.full_sql,
                refined_sql=refined_sql,
                success=True
            )

        except Exception as e:
            return RefinementResult(
                original_sql=context.full_sql,
                refined_sql=context.full_sql,
                success=False,
                error_message=str(e)
            )

    def _build_prompt(self, context: OrderedRefinementContext) -> str:
        errors_section = []

        for i, ec in enumerate(context.error_contexts, 1):
            error_block = f"""### Error {i}: {ec.error_type}

**Error Location (AST Subtree)**:
```
{ec.subtree}
```

**Relevant Schema Information**:
{ec.subgraph if ec.subgraph else "Not applicable for this error type"}

**Error Analysis**:
{self._format_guideline(ec.filled_guideline)}

**Reference Example**:
{ec.few_shot}

---"""
            errors_section.append(error_block)

        prompt = f"""## Task
You are an SQL Refiner. Fix ALL the errors listed below in the SQL query.
Process the errors in the order presented (they are sorted by priority).

## Current SQL
```sql
{context.full_sql}
```

## Errors to Fix (in order of priority)

{chr(10).join(errors_section)}

## Instructions
1. Fix ALL errors listed above
2. Process them in order - earlier errors may affect later fixes
3. Output the COMPLETE fixed SQL query
4. Do NOT add any explanations, just output the SQL

## Fixed SQL
```sql
"""

        return prompt

    def _format_guideline(self, guideline: dict) -> str:
        if not guideline:
            return "No specific analysis available."

        lines = []
        for key, value in guideline.items():
            readable_key = key.replace("_", " ").title()
            lines.append(f"- **{readable_key}**: {value}")

        return "\n".join(lines)

    def _extract_sql(self, response_text: str) -> str:
        sql_match = re.search(r'```sql\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()

        code_match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        select_match = re.search(r'(SELECT\s+.*)', response_text, re.DOTALL | re.IGNORECASE)
        if select_match:
            sql = select_match.group(1).strip()
            if '\n\n' in sql:
                sql = sql.split('\n\n')[0]
            return sql

        return response_text.strip()

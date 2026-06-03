import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .types import ErrorAnalysis, RefinementInput, ERROR_DESCRIPTIONS, ERROR_PRIORITY
from .prompts import GUIDELINE_TEMPLATES, get_template_fields


class LocLLM:

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None, temperature: Optional[float] = None):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def analyze(self, refinement_input: RefinementInput) -> List[ErrorAnalysis]:
        if not refinement_input.error_types:
            return []

        prompt = self._build_prompt(refinement_input)

        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an SQL Error Analyzer. Analyze SQL errors precisely and fill in guideline templates."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }

        response = self.client.chat.completions.create(**kwargs)

        result_text = response.choices[0].message.content
        return self._parse_response(result_text, refinement_input.error_types)

    def _build_prompt(self, inp: RefinementInput) -> str:
        error_desc_list = []
        for i, error_type in enumerate(inp.error_types, 1):
            desc = ERROR_DESCRIPTIONS.get(error_type, "")
            error_desc_list.append(f"{i}. {error_type}: {desc}")

        templates_section = []
        for error_type in inp.error_types:
            template = GUIDELINE_TEMPLATES.get(error_type, "")
            if template:
                fields = get_template_fields(error_type)
                templates_section.append(f"### Template for {error_type}\nFields to fill: {fields}\n")

        prompt = f"""## Task
You are an SQL Error Analyzer. For each detected error type, you must:
1. Locate the exact AST node(s) where the error occurs
2. Identify the schema elements (tables/columns) involved
3. Fill in the Guideline template with your case-specific analysis

## Input

**Question**: {inp.question}

**External Knowledge**: {inp.external_knowledge or "None"}

**Schema Structure**:
[Nodes]
{inp.schema_structure_nodes}

[Edges]
{inp.schema_structure_edges}

**Predicted SQL**:
{inp.predicted_sql}

**AST**:
{inp.ast}

**Detected Errors**: 
{chr(10).join(error_desc_list)}

## Guideline Templates to Fill

{chr(10).join(templates_section)}

## Output Format

Output a JSON object with an "analyses" array containing one object per error:
```json
{{
  "analyses": [
    {{
      "error_type": "Error Type Name",
      "located_nodes": ["Node[i]", "Node[j]"],
      "involved_elements": ["table.column", ...],
      "filled_guideline": {{
        // template-specific fields based on error type
      }}
    }}
  ]
}}
```

Important:
- located_nodes should reference specific Node[n] entries from the AST
- involved_elements should list relevant table.column or table names
- filled_guideline should contain the key fields needed to fix this error

Analyze each error and output the JSON:"""

        return prompt

    def _parse_response(self, response_text: str, error_types: List[str]) -> List[ErrorAnalysis]:
        try:
            data = json.loads(response_text)
            analyses = data.get("analyses", [])

            result = []
            for analysis in analyses:
                error_analysis = ErrorAnalysis(
                    error_type=analysis.get("error_type", ""),
                    located_nodes=analysis.get("located_nodes", []),
                    involved_elements=analysis.get("involved_elements", []),
                    filled_guideline=analysis.get("filled_guideline", {})
                )
                result.append(error_analysis)

            return result

        except json.JSONDecodeError:
            return [
                ErrorAnalysis(
                    error_type=et,
                    located_nodes=[],
                    involved_elements=[],
                    filled_guideline={"error": "Failed to parse LLM response"}
                )
                for et in error_types
            ]

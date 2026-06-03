import json
import re
import argparse
import os
from typing import Any, Dict, List, Optional

from .loc_llm import LocLLM
from .context_builder import ContextBuilder
from .types import (
    ErrorAnalysis,
    OrderedRefinementContext,
    RefinementInput,
    RefinementResult,
    ERROR_TOKEN_TO_NAME,
    parse_error_tokens,
)
from .ref_llm import RefLLM


class RefinementPipeline:

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        verbose: bool = False,
        temperature: Optional[float] = None
    ):
        self.loc_llm = LocLLM(model=model, api_key=api_key, temperature=temperature)
        self.context_builder = ContextBuilder()
        self.ref_llm = RefLLM(model=model, api_key=api_key, temperature=temperature)
        self.verbose = verbose

    def refine(self, refinement_input: RefinementInput) -> RefinementResult:
        if self.verbose:
            print(f"[Pipeline] Starting refinement for {len(refinement_input.error_types)} errors")
            print(f"[Pipeline] Error types: {refinement_input.error_types}")

        if self.verbose:
            print("[Pipeline] Step 1: Analyzing errors with LocLLM...")

        error_analyses = self.loc_llm.analyze(refinement_input)

        if self.verbose:
            print(f"[Pipeline] LocLLM produced {len(error_analyses)} analyses")

        if self.verbose:
            print("[Pipeline] Step 2: Building refinement context...")

        context = self.context_builder.build(refinement_input, error_analyses)

        if self.verbose:
            print(f"[Pipeline] Context built with {len(context.error_contexts)} error contexts")

        if self.verbose:
            print("[Pipeline] Step 3: Refining SQL with RefLLM...")

        result = self.ref_llm.refine(context)
        result.error_analyses = error_analyses

        if self.verbose:
            print(f"[Pipeline] Refinement complete. Success: {result.success}")
            if result.success:
                print(f"[Pipeline] Original SQL: {result.original_sql[:100]}...")
                print(f"[Pipeline] Refined SQL: {result.refined_sql[:100]}...")

        return result

    def refine_from_sample(self, sample: Dict[str, Any]) -> RefinementResult:
        output_str = sample.get("output", "")

        if output_str.strip() == "<no_error>":
            input_text = sample.get("input", "")
            sql = self._extract_sql_from_input(input_text)
            return RefinementResult(
                original_sql=sql,
                refined_sql=sql,
                success=True
            )

        refinement_input = self._parse_sample(sample)
        return self.refine(refinement_input)

    def _parse_sample(self, sample: Dict[str, Any]) -> RefinementInput:
        input_text = sample.get("input", "")
        output_str = sample.get("output", "")

        tokens = parse_error_tokens(output_str)
        error_types = [ERROR_TOKEN_TO_NAME.get(t, t) for t in tokens]

        sections = self._parse_input_sections(input_text)

        return RefinementInput(
            question=sections.get("question", ""),
            external_knowledge=sections.get("external_knowledge", ""),
            schema_structure_nodes=sections.get("nodes", ""),
            schema_structure_edges=sections.get("edges", ""),
            predicted_sql=sections.get("sql", ""),
            ast=sections.get("ast", ""),
            error_types=error_types,
            rule_based_results=sections.get("rule_based", None),
            execution_results=sections.get("execution", None),
        )

    def _parse_input_sections(self, input_text: str) -> Dict[str, str]:
        sections = {}
        current_section = None
        current_content = []

        lines = input_text.split("\n")

        for line in lines:
            if line.startswith("[Question]"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "question"
                current_content = []
            elif line.startswith("[External Knowledge]"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "external_knowledge"
                current_content = []
            elif line.startswith("[Question-Schema Structure]"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "schema"
                current_content = []
            elif line.startswith("[Nodes]"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "nodes"
                current_content = []
            elif line.startswith("[Edges]"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "edges"
                current_content = []
            elif line.startswith("[Predicted SQL]"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "sql"
                current_content = []
            elif line.startswith("[SQL Abstract Syntax Tree]"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "ast"
                current_content = []
            elif line.startswith("[Rule-based Detection Results]"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "rule_based"
                current_content = []
            elif line.startswith("[Execution Results]"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = "execution"
                current_content = []
            else:
                current_content.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def _extract_sql_from_input(self, input_text: str) -> str:
        sections = self._parse_input_sections(input_text)
        return sections.get("sql", "")


def load_samples_with_errors(filepath: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    with open(filepath, "r") as f:
        data = json.load(f)

    samples_with_errors = [
        sample for sample in data
        if sample.get("output", "").strip() != "<no_error>"
    ]

    if limit:
        samples_with_errors = samples_with_errors[:limit]

    return samples_with_errors


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o")
    parser.add_argument("--limit", "-n", type=int, default=5)
    parser.add_argument("--model", "-m", default="gpt-4o")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    print(f"Loading samples from {args.input}...")
    samples = load_samples_with_errors(args.input, limit=args.limit)
    print(f"Found {len(samples)} samples with errors")

    api_key = os.environ.get("OPENAI_API_KEY")
    pipeline = RefinementPipeline(model=args.model, api_key=api_key, verbose=args.verbose)

    results = []
    for i, sample in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"Processing sample {i+1}/{len(samples)}")
        print(f"Errors: {sample.get('output', '')}")

        result = pipeline.refine_from_sample(sample)

        results.append({
            "original_sql": result.original_sql,
            "refined_sql": result.refined_sql,
            "success": result.success,
            "error_message": result.error_message,
            "detected_errors": sample.get("output", ""),
        })

        print(f"Original: {result.original_sql}")
        print(f"Refined:  {result.refined_sql}")
        print(f"Success:  {result.success}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()

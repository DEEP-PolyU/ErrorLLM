import re
from typing import Dict


GUIDELINE_TEMPLATES: Dict[str, str] = {
    "Attribute Mismatch": """## Error: Attribute Mismatch
A wrong column is used in the SQL.

### Localization
- **Error Node(s)**: {nodes}
- **Current Column**: {current_column}
- **Located In Clause**: {clause}

### Semantic Analysis
- **NL Requirement**: {what_nl_asks_for}
- **Why Current is Wrong**: {why_wrong}
- **External Knowledge Hint**: {hint_if_any}

### Candidates
- **Candidate Columns**: {candidate_columns}
- **Recommended Fix**: {recommended_column}

### Fix Instruction
Replace `{current_column}` with `{recommended_column}` because {reason}.""",

    "Attribute Redundancy": """## Error: Attribute Redundancy
An unnecessary column is selected that's not required by the question.

### Localization
- **Error Node(s)**: {nodes}
- **Redundant Column**: {redundant_column}

### Analysis
- **NL Required Outputs**: {what_nl_asks_for}
- **Why Redundant**: {why_not_needed}

### Fix Instruction
Remove `{redundant_column}` from SELECT clause.""",

    "Attribute Missing": """## Error: Attribute Missing
A required column is not included in the SQL.

### Localization
- **Missing From Clause**: {clause}
- **Current Columns**: {current_columns}

### Analysis
- **NL Requirement**: {what_nl_asks_for}
- **Missing Column**: {missing_column}
- **Column's Table**: {column_table}

### Fix Instruction
Add `{missing_column}` to {clause} clause.""",

    "Table Mismatch": """## Error: Table Mismatch
A wrong table is used in the SQL.

### Localization
- **Error Node(s)**: {nodes}
- **Current Table**: {current_table}

### Analysis
- **NL Requirement**: {what_entity_needed}
- **Why Current is Wrong**: {why_wrong}
- **Correct Table**: {correct_table}

### Fix Instruction
Replace `{current_table}` with `{correct_table}`. Update JOIN conditions accordingly.""",

    "Table Redundancy": """## Error: Table Redundancy
An unnecessary table is joined that doesn't contribute to the query.

### Localization
- **Error Node(s)**: {nodes}
- **Redundant Table**: {redundant_table}

### Analysis
- **Columns Used from This Table**: {columns_used}
- **Why Redundant**: {why_not_needed}

### Fix Instruction
Remove `{redundant_table}` and its JOIN clause from the query.""",

    "Table Missing": """## Error: Table Missing
A required table is not joined in the FROM clause.

### Localization
- **Current Tables**: {current_tables}
- **Problem Signal**: {signal}

### Analysis
- **Missing Table**: {missing_table}
- **Why Needed**: {why_needed}
- **Join Path**: {join_path}
- **Join Condition**: {join_condition}

### Fix Instruction
Add JOIN: `INNER JOIN {missing_table} ON {join_condition}`""",

    "Value Mismatch": """## Error: Value Mismatch
A literal value in the SQL is incorrect or has wrong format.

### Localization
- **Error Node(s)**: {nodes}
- **Current Value**: {current_value}
- **Located In**: {clause}

### Analysis
- **NL Specified Value**: {correct_value_from_nl}
- **Column Data Type**: {data_type}
- **Format Issue**: {format_issue_if_any}

### Fix Instruction
Replace `{current_value}` with `{correct_value}`.""",

    "Condition Missing": """## Error: Condition Missing
A required filter condition is not included in WHERE/HAVING.

### Localization
- **Should Add To**: {where_or_having}
- **Current Conditions**: {current_conditions}

### Analysis
- **NL Constraint**: {constraint_from_nl}
- **External Knowledge**: {implicit_condition_hint}
- **Missing Condition**: {missing_condition}

### Fix Instruction
Add condition: `{missing_condition}` to {where_or_having} clause.""",

    "Condition Error": """## Error: Condition Error
A condition in WHERE/HAVING is incorrect or redundant.

### Localization
- **Error Node(s)**: {nodes}
- **Problematic Condition**: {condition}

### Analysis
- **Issue Type**: {mismatch_or_redundant}
- **What's Wrong**: {explanation}

### Fix Instruction
{specific_fix_instruction}""",

    "Function Error": """## Error: Function Error
An aggregate, date, string, or conditional function is used incorrectly.

### Localization
- **Error Node(s)**: {nodes}
- **Current Function**: {current_function}
- **Function Arguments**: {arguments}

### Analysis
- **NL Requirement**: {what_nl_asks_for}
- **Function Category**: {aggregate_date_string_conditional}
- **What's Wrong**: {explanation}
- **Correct Function**: {correct_function}

### Fix Instruction
Replace `{current_function}({arguments})` with `{correct_function}({correct_arguments})`.""",

    "Clause Error": """## Error: Clause Error
A structural clause (GROUP BY, ORDER BY, LIMIT) is missing or redundant.

### Localization
- **Issue Type**: {missing_or_redundant}
- **Affected Clause**: {clause_type}

### Analysis
- **NL Signals**: {signals_from_nl}
- **Why Needed/Not Needed**: {explanation}

### Fix Instruction
{add_or_remove} {clause_type} clause: `{clause_content}`.""",

    "DISTINCT Error": """## Error: DISTINCT Error
DISTINCT keyword is either missing or incorrectly applied.

### Localization
- **Current State**: {has_distinct_or_not}
- **Should Have DISTINCT**: {yes_or_no}

### Analysis
- **NL Signals**: {signals_for_uniqueness}
- **Why Change**: {explanation}

### Fix Instruction
{add_or_remove} DISTINCT keyword in SELECT clause.""",
}


FEW_SHOT_EXAMPLES: Dict[str, Dict[str, str]] = {
    "Attribute Mismatch": {
        "question": "What are the names of products with price over 100?",
        "wrong_sql": "SELECT product_id FROM products WHERE price > 100",
        "error_description": "Selected product_id instead of product_name. NL asks for 'names'.",
        "fixed_sql": "SELECT product_name FROM products WHERE price > 100",
    },
    "Attribute Redundancy": {
        "question": "List all customer names",
        "wrong_sql": "SELECT customer_id, customer_name, email FROM customers",
        "error_description": "customer_id and email not requested in the question.",
        "fixed_sql": "SELECT customer_name FROM customers",
    },
    "Attribute Missing": {
        "question": "Show employee names and their department names",
        "wrong_sql": "SELECT emp_name FROM employees JOIN departments ON employees.dept_id = departments.id",
        "error_description": "Missing dept_name in SELECT clause.",
        "fixed_sql": "SELECT emp_name, dept_name FROM employees JOIN departments ON employees.dept_id = departments.id",
    },
    "Table Mismatch": {
        "question": "Count total sales amount",
        "wrong_sql": "SELECT SUM(amount) FROM orders",
        "error_description": "Should use sales table, not orders table.",
        "fixed_sql": "SELECT SUM(amount) FROM sales",
    },
    "Table Redundancy": {
        "question": "List all product names",
        "wrong_sql": "SELECT product_name FROM products JOIN categories ON products.category_id = categories.id",
        "error_description": "categories table is not needed for this query.",
        "fixed_sql": "SELECT product_name FROM products",
    },
    "Table Missing": {
        "question": "Find customer names who placed orders",
        "wrong_sql": "SELECT customer_name FROM orders",
        "error_description": "customer_name is in customers table, need JOIN.",
        "fixed_sql": "SELECT customer_name FROM orders JOIN customers ON orders.customer_id = customers.customer_id",
    },
    "Value Mismatch": {
        "question": "Find orders from year 2023",
        "wrong_sql": "SELECT * FROM orders WHERE year = '2022'",
        "error_description": "Wrong year value (2022 instead of 2023).",
        "fixed_sql": "SELECT * FROM orders WHERE year = '2023'",
    },
    "Condition Missing": {
        "question": "Find active employees with salary over 50000",
        "external_knowledge": "active employees have status = 'active'",
        "wrong_sql": "SELECT * FROM employees WHERE salary > 50000",
        "error_description": "Missing status = 'active' condition from external knowledge.",
        "fixed_sql": "SELECT * FROM employees WHERE salary > 50000 AND status = 'active'",
    },
    "Condition Error": {
        "question": "Find products cheaper than 50",
        "wrong_sql": "SELECT * FROM products WHERE price > 50",
        "error_description": "Wrong operator (> instead of <).",
        "fixed_sql": "SELECT * FROM products WHERE price < 50",
    },
    "Function Error": {
        "question": "What is the total quantity sold?",
        "wrong_sql": "SELECT COUNT(quantity) FROM sales",
        "error_description": "Should use SUM for total, not COUNT.",
        "fixed_sql": "SELECT SUM(quantity) FROM sales",
    },
    "Clause Error": {
        "question": "Find the top 3 highest paid employees",
        "wrong_sql": "SELECT name, salary FROM employees",
        "error_description": "Missing ORDER BY and LIMIT clauses.",
        "fixed_sql": "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 3",
    },
    "Modifier Error": {
        "question": "List all unique categories",
        "wrong_sql": "SELECT category FROM products",
        "error_description": "Missing DISTINCT for 'unique' requirement.",
        "fixed_sql": "SELECT DISTINCT category FROM products",
    },
}


def get_template(error_type: str) -> str:
    return GUIDELINE_TEMPLATES.get(error_type, "")


def get_template_fields(error_type: str) -> list:
    template = GUIDELINE_TEMPLATES.get(error_type, "")
    return re.findall(r'\{(\w+)\}', template)


def get_few_shot(error_type: str) -> Dict[str, str]:
    return FEW_SHOT_EXAMPLES.get(error_type, {})


def format_few_shot_for_prompt(error_type: str) -> str:
    example = FEW_SHOT_EXAMPLES.get(error_type, {})
    if not example:
        return "No reference example available."
    lines = [
        f"- Wrong: `{example.get('wrong_sql', '')}`",
        f"- Error: {example.get('error_description', '')}",
        f"- Fixed: `{example.get('fixed_sql', '')}`",
    ]
    return "\n".join(lines)

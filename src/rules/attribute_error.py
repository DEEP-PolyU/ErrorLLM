import re
from typing import Dict, List, Set
from .base import ASTNode, QSSParser, ErrorLabels, BaseDetector, normalize_name


EXEC_COLUMN_PATTERNS = [
    (re.compile(r"no such column:\s*([^\s,\)]+)", re.IGNORECASE), ErrorLabels.ATTRIBUTE_REDUNDANCY),
    (re.compile(r"ambiguous column name:\s*([^\s,\)]+)", re.IGNORECASE), ErrorLabels.ATTRIBUTE_MISMATCH),
    (re.compile(r"misuse of aliased aggregate\s+([^\s]+)", re.IGNORECASE), ErrorLabels.ATTRIBUTE_MISMATCH),
]


def fuzzy_column_match(col_name: str, schema_names: Set[str]) -> bool:
    col_norm = normalize_name(col_name)
    if not col_norm:
        return True
    
    if col_norm in schema_names:
        return True
    
    for schema_col in schema_names:
        if col_norm in schema_col or schema_col in col_norm:
            return True
        col_words = set(col_norm.replace('_', ' ').split())
        schema_words = set(schema_col.replace('_', ' ').split())
        common_words = col_words & schema_words
        trivial_words = {'id', 'name', 'type', 'date', 'code', 'num', 'count', 'the', 'a', 'of'}
        significant_common = common_words - trivial_words
        if significant_common:
            return True
    
    return False


class AttributeErrorDetector(BaseDetector):
    def detect(self, ast: ASTNode, qss: QSSParser, question: str, **kwargs) -> List[Dict]:
        errors = []
        
        ast_columns = ast.get_all_column_names()
        
        schema_columns = qss.get_all_schema_columns()
        schema_col_names_normalized = set()
        
        for col_id in schema_columns:
            col_name = col_id.split('.')[-1] if '.' in col_id else col_id
            schema_col_names_normalized.add(normalize_name(col_name))
            col_info = qss.get_column_info(col_id)
            if col_info and col_info.get('label'):
                schema_col_names_normalized.add(normalize_name(col_info['label']))
        
        seen_extra = set()
        for col in ast_columns:
            col_norm = normalize_name(col)
            if col_norm and col_norm not in seen_extra:
                if not fuzzy_column_match(col, schema_col_names_normalized):
                    seen_extra.add(col_norm)
                    errors.append({
                        'error_label': ErrorLabels.ATTRIBUTE_REDUNDANCY,
                        'column': col,
                        'target_node_type': 'column',
                        'target_value': col
                    })
        
        exec_error = kwargs.get('exec_error', '')
        if exec_error:
            exec_errors = self._detect_from_exec_error(exec_error)
            errors.extend(exec_errors)
        
        return errors
    
    def _detect_from_exec_error(self, error_message: str) -> List[Dict]:
        errors = []
        for pattern, error_label in EXEC_COLUMN_PATTERNS:
            match = pattern.search(error_message)
            if match:
                col_name = match.group(1).strip('"`\'') if match.lastindex else None
                error_dict = {
                    'error_label': error_label,
                    'source': 'execution_error',
                    'error_message': error_message,
                }
                if col_name:
                    error_dict['column'] = col_name
                    error_dict['target_node_type'] = 'column'
                    error_dict['target_value'] = col_name
                errors.append(error_dict)
                break
        return errors

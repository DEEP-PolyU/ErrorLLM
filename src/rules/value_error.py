import re
from typing import Dict, List
from .base import ASTNode, QSSParser, ErrorLabels, BaseDetector, normalize_name


EXEC_VALUE_PATTERNS = [
    (re.compile(r"datatype mismatch", re.IGNORECASE), ErrorLabels.VALUE_ERROR),
    (re.compile(r"cannot compare", re.IGNORECASE), ErrorLabels.VALUE_ERROR),
    (re.compile(r"invalid.*value", re.IGNORECASE), ErrorLabels.VALUE_ERROR),
    (re.compile(r"cannot (?:convert|cast)", re.IGNORECASE), ErrorLabels.VALUE_ERROR),
]


class ValueErrorDetector(BaseDetector):
    
    def detect(self, ast: ASTNode, qss: QSSParser, question: str, **kwargs) -> List[Dict]:
        errors = []
        
        where_node = ast.get_where_node()
        if not where_node:
            return errors
        
        literals = where_node.get_all_literals()
        
        for value, value_type in literals:
            if value_type == 'string' and value is not None:
                str_val = str(value).strip()
                if str_val.isdigit() or (str_val.replace('.', '', 1).isdigit() and str_val.count('.') <= 1):
                    pass
        
        exec_error = kwargs.get('exec_error', '')
        if exec_error:
            exec_errors = self._detect_from_exec_error(exec_error)
            errors.extend(exec_errors)
        
        return errors
    
    def _detect_from_exec_error(self, error_message: str) -> List[Dict]:
        errors = []
        for pattern, error_label in EXEC_VALUE_PATTERNS:
            match = pattern.search(error_message)
            if match:
                errors.append({
                    'error_label': error_label,
                    'source': 'execution_error',
                    'error_message': error_message,
                })
                break
        return errors

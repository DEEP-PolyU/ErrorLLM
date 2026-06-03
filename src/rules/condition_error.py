import re
from typing import Dict, List
from .base import ASTNode, QSSParser, ErrorLabels, BaseDetector, normalize_name


EXEC_CONDITION_PATTERNS = [
    (re.compile(r'near "(?:AND|OR)":\s*syntax error', re.IGNORECASE), ErrorLabels.CONDITION_ERROR),
    (re.compile(r"BETWEEN.*requires", re.IGNORECASE), ErrorLabels.CONDITION_MISSING),
    (re.compile(r"IN requires", re.IGNORECASE), ErrorLabels.CONDITION_ERROR),
]


class ConditionErrorDetector(BaseDetector):
    
    def detect(self, ast: ASTNode, qss: QSSParser, question: str, **kwargs) -> List[Dict]:
        errors = []

        exec_error = kwargs.get('exec_error', '')
        if exec_error:
            exec_errors = self._detect_from_exec_error(exec_error)
            errors.extend(exec_errors)
        
        return errors
    
    def _detect_from_exec_error(self, error_message: str) -> List[Dict]:
        
        errors = []
        for pattern, error_label in EXEC_CONDITION_PATTERNS:
            match = pattern.search(error_message)
            if match:
                errors.append({
                    'error_label': error_label,
                    'source': 'execution_error',
                    'error_message': error_message,
                })
                break
        return errors

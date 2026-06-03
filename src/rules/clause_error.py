
import re
from typing import Dict, List
from .base import ASTNode, QSSParser, ErrorLabels, BaseDetector



EXEC_CLAUSE_PATTERNS = [
    
    (re.compile(r"must appear in the GROUP BY clause", re.IGNORECASE), "groupby"),
    (re.compile(r"not in GROUP BY", re.IGNORECASE), "groupby"),
    (re.compile(r"SELECT list.*not in GROUP BY", re.IGNORECASE), "groupby"),
    
    (re.compile(r"ORDER BY term.*does not match", re.IGNORECASE), "order"),
    (re.compile(r"ORDER BY position.*not in", re.IGNORECASE), "order"),
]


class ClauseErrorDetector(BaseDetector):
    
    def detect(self, ast: ASTNode, qss: QSSParser, question: str, **kwargs) -> List[Dict]:
        errors = []
        exec_error = kwargs.get('exec_error', '')
        if exec_error:
            errors.extend(self._detect_from_exec_error(exec_error))
        
        return errors
    
    def _detect_from_exec_error(self, error_message: str) -> List[Dict]:
        
        errors = []
        for pattern, node_type in EXEC_CLAUSE_PATTERNS:
            if pattern.search(error_message):
                errors.append({
                    'error_label': ErrorLabels.CLAUSE_ERROR,
                    'source': 'execution_error',
                    'error_message': error_message,
                    'target_node_type': node_type,
                })
                break
        return errors

import re
from typing import Dict, List
from .base import ASTNode, QSSParser, ErrorLabels, BaseDetector, normalize_name


INVALID_SQLITE_FUNCTIONS = {
    'YEAR', 'MONTH', 'DAY', 'DATEDIFF', 'NOW', 'CURDATE', 'GETDATE',
    'DATEADD', 'DATENAME', 'DATEPART', 'ISNULL', 'NVL', 'CONVERT'
}

EXEC_FUNCTION_PATTERNS = [
    (re.compile(r"no such function:\s*([^\s\(]+)", re.IGNORECASE), ErrorLabels.FUNCTION_ERROR),
    (re.compile(r"wrong number of arguments to function\s+([^\s\(]+)", re.IGNORECASE), ErrorLabels.FUNCTION_ERROR),
    (re.compile(r"misuse of aggregate function\s*:?\s*([^\s\(]+)?", re.IGNORECASE), ErrorLabels.FUNCTION_ERROR),
    (re.compile(r"misuse of aggregate:\s*([^\s\(]+)", re.IGNORECASE), ErrorLabels.FUNCTION_ERROR),
    (re.compile(r"aggregate functions? in aggregate", re.IGNORECASE), ErrorLabels.FUNCTION_ERROR),
]


class FunctionErrorDetector(BaseDetector):
    
    def detect(self, ast: ASTNode, qss: QSSParser, question: str, **kwargs) -> List[Dict]:
        errors = []
        
        func_nodes = ast.find_nodes_by_types({'func', 'function', 'call'})
        for func_node in func_nodes:
            func_name = str(func_node.name or '').upper()
            if func_name in INVALID_SQLITE_FUNCTIONS:
                errors.append({
                    'error_label': ErrorLabels.FUNCTION_ERROR,
                    'function': func_name,
                    'target_node_type': func_node.node_type,
                    'target_value': func_node.name
                })
        
        
        exec_error = kwargs.get('exec_error', '')
        if exec_error:
            exec_errors = self._detect_from_exec_error(exec_error)
            errors.extend(exec_errors)
        
        return errors
    
    def _detect_from_exec_error(self, error_message: str) -> List[Dict]:
        errors = []
        for pattern, error_label in EXEC_FUNCTION_PATTERNS:
            match = pattern.search(error_message)
            if match:
                error_dict = {
                    'error_label': error_label,
                    'source': 'execution_error',
                    'error_message': error_message,
                }
                if match.lastindex and match.lastindex >= 1:
                    func_name = match.group(1)
                    if func_name:
                        func_name = func_name.strip('"`\'')
                        error_dict['function'] = func_name
                        error_dict['target_node_type'] = 'function'
                        error_dict['target_value'] = func_name
                errors.append(error_dict)
                break
        return errors

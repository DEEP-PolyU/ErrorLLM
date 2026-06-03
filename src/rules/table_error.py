import re
from typing import Dict, List, Set
from .base import ASTNode, QSSParser, ErrorLabels, BaseDetector, normalize_name


EXEC_TABLE_PATTERNS = [
    (re.compile(r"no such table:\s*([^\s,\)]+)", re.IGNORECASE), ErrorLabels.TABLE_REDUNDANCY),
    (re.compile(r"no tables specified", re.IGNORECASE), ErrorLabels.TABLE_MISSING),
]


def fuzzy_table_match(table_name: str, schema_names: Set[str]) -> bool:
    table_norm = normalize_name(table_name)
    if not table_norm:
        return True
    
    if table_norm in schema_names:
        return True
    
    for schema_table in schema_names:
        if table_norm in schema_table or schema_table in table_norm:
            return True
    
    return False


class TableErrorDetector(BaseDetector):
    
    def detect(self, ast: ASTNode, qss: QSSParser, question: str, **kwargs) -> List[Dict]:
        errors = []
        
        ast_tables = ast.get_all_table_names()
        
        schema_tables = qss.get_all_schema_tables()
        
        schema_table_names_normalized = set()
        for table_id in schema_tables:
            schema_table_names_normalized.add(normalize_name(table_id))
            table_info = qss.node_by_id.get(table_id, {})
            if table_info.get('label'):
                schema_table_names_normalized.add(normalize_name(table_info['label']))
        
        seen_extra = set()
        for table in ast_tables:
            table_norm = normalize_name(table)
            if table_norm and table_norm not in seen_extra:
                if not fuzzy_table_match(table, schema_table_names_normalized):
                    seen_extra.add(table_norm)
                    errors.append({
                        'error_label': ErrorLabels.TABLE_REDUNDANCY,
                        'table': table,
                        'target_node_type': 'table',
                        'target_value': table
                    })
        
        
        exec_error = kwargs.get('exec_error', '')
        if exec_error:
            exec_errors = self._detect_from_exec_error(exec_error)
            errors.extend(exec_errors)
        
        return errors
    
    def _detect_from_exec_error(self, error_message: str) -> List[Dict]:
        errors = []
        for pattern, error_label in EXEC_TABLE_PATTERNS:
            match = pattern.search(error_message)
            if match:
                error_dict = {
                    'error_label': error_label,
                    'source': 'execution_error',
                    'error_message': error_message,
                }
                if match.lastindex and match.lastindex >= 1:
                    table_name = match.group(1).strip('"`\'')
                    error_dict['table'] = table_name
                    error_dict['target_node_type'] = 'table'
                    error_dict['target_value'] = table_name
                errors.append(error_dict)
                break
        return errors

from typing import Dict, List, Optional, Set, Tuple
import re


class ErrorLabels:
    ATTRIBUTE_MISMATCH = "attribute_mismatch"

    ATTRIBUTE_REDUNDANCY = "attribute_redundancy"

    ATTRIBUTE_MISSING = "attribute_missing"

    TABLE_MISMATCH = "table_mismatch"

    TABLE_REDUNDANCY = "table_redundancy"

    TABLE_MISSING = "table_missing"

    VALUE_ERROR = "value_error"

    CONDITION_MISSING = "condition_missing"

    CONDITION_ERROR = "condition_error"

    FUNCTION_ERROR = "function_error"

    CLAUSE_ERROR = "clause_error"

    MODIFIER_ERROR = "modifier_error"

    NO_ERROR = "no_error"


def normalize_name(name: str) -> str:

    if name is None:
        return ""
    return str(name).lower().replace(' ', '_').replace('-', '_').replace('`', '').replace('"', '').strip()


class ASTNode:
    
    def __init__(self, node_dict: Dict):
        self.node_type = node_dict.get('node_type', '')
        self.name = node_dict.get('name')
        self.value = node_dict.get('value')
        self.value_type = node_dict.get('value_type')
        self.children = [ASTNode(c) for c in node_dict.get('children', [])]
        self._raw = node_dict
    
    @classmethod
    def from_dict(cls, node_dict: Dict) -> 'ASTNode':
        return cls(node_dict)
    
    def find_nodes_by_type(self, node_type: str) -> List['ASTNode']:
        
        results = []
        if self.node_type == node_type:
            results.append(self)
        for child in self.children:
            results.extend(child.find_nodes_by_type(node_type))
        return results
    
    def find_nodes_by_types(self, node_types: Set[str]) -> List['ASTNode']:
        
        results = []
        if self.node_type in node_types:
            results.append(self)
        for child in self.children:
            results.extend(child.find_nodes_by_types(node_types))
        return results
    
    def get_all_column_names(self) -> List[str]:
        
        columns = []
        if self.node_type == 'column' and self.name:
            columns.append(self.name)
        for child in self.children:
            columns.extend(child.get_all_column_names())
        return columns
    
    def get_all_table_names(self) -> List[str]:
        
        tables = []
        if self.node_type == 'table' and self.name:
            tables.append(self.name)
        for child in self.children:
            tables.extend(child.get_all_table_names())
        return tables
    
    def get_all_literals(self) -> List[Tuple[any, str]]:
        
        literals = []
        if self.node_type == 'literal':
            literals.append((self.value, self.value_type))
        for child in self.children:
            literals.extend(child.get_all_literals())
        return literals
    
    def has_aggregation(self) -> bool:
        
        agg_types = {'count', 'sum', 'avg', 'max', 'min'}
        return len(self.find_nodes_by_types(agg_types)) > 0
    
    def get_select_columns(self) -> List['ASTNode']:
        
        columns = []
        for child in self.children:
            if child.node_type in ('from', 'where', 'join', 'groupby', 'order', 'limit', 'having'):
                continue
            columns.extend(child.find_nodes_by_type('column'))
        return columns
    
    def get_where_node(self) -> Optional['ASTNode']:
        
        where_nodes = self.find_nodes_by_type('where')
        return where_nodes[0] if where_nodes else None
    
    def get_from_node(self) -> Optional['ASTNode']:
        
        from_nodes = self.find_nodes_by_type('from')
        return from_nodes[0] if from_nodes else None
    
    def get_join_nodes(self) -> List['ASTNode']:
        
        return self.find_nodes_by_type('join')
    
    def get_groupby_node(self) -> Optional['ASTNode']:
        
        nodes = self.find_nodes_by_type('groupby')
        return nodes[0] if nodes else None
    
    def get_orderby_node(self) -> Optional['ASTNode']:
        
        nodes = self.find_nodes_by_type('order')
        return nodes[0] if nodes else None
    
    def get_limit_node(self) -> Optional['ASTNode']:
        
        nodes = self.find_nodes_by_type('limit')
        return nodes[0] if nodes else None
    
    def has_distinct(self) -> bool:
        
        return len(self.find_nodes_by_type('distinct')) > 0
    
    def get_comparison_nodes(self) -> List['ASTNode']:
        
        cmp_types = {'eq', 'neq', 'gt', 'lt', 'gte', 'lte', 'like', 'in', 'between', 'is', 'is_not'}
        return self.find_nodes_by_types(cmp_types)
    
    def get_logical_nodes(self) -> List['ASTNode']:
        
        return self.find_nodes_by_types({'and', 'or'})


class QSSParser:

    def __init__(self, qss_dict: Dict):
        self.db = qss_dict.get('db', '')
        self.nodes = qss_dict.get('nodes', [])
        self.edges = qss_dict.get('edges', [])
        self._build_indexes()
    
    def _build_indexes(self):
        
        self.node_by_id = {n['id']: n for n in self.nodes}
        self.tables = {n['id']: n for n in self.nodes if n.get('type') == 'table'}
        self.columns = {n['id']: n for n in self.nodes if n.get('type') == 'column'}
        self.question_entities = {n['id']: n for n in self.nodes if n.get('type') == 'question_entity'}
        
        
        self.edges_by_src = {}
        self.edges_by_tgt = {}
        for edge in self.edges:
            src, tgt = edge['src'], edge['tgt']
            if src not in self.edges_by_src:
                self.edges_by_src[src] = []
            self.edges_by_src[src].append(edge)
            if tgt not in self.edges_by_tgt:
                self.edges_by_tgt[tgt] = []
            self.edges_by_tgt[tgt].append(edge)
    
    def get_required_columns(self) -> Set[str]:
        
        required = set()
        for edge in self.edges:
            if edge['rel'] == 'relates_to_column' and edge['src'].startswith('q_ent_'):
                required.add(edge['tgt'])
        return required
    
    def get_required_tables(self) -> Set[str]:
        
        required = set()
        for edge in self.edges:
            if edge['rel'] == 'relates_to_table' and edge['src'].startswith('q_ent_'):
                required.add(edge['tgt'])
        return required
    
    def get_all_schema_columns(self) -> Set[str]:
        
        return set(self.columns.keys())
    
    def get_all_schema_tables(self) -> Set[str]:
        
        return set(self.tables.keys())
    
    def get_column_info(self, col_id: str) -> Optional[Dict]:
        
        return self.columns.get(col_id)
    
    def get_table_for_column(self, col_id: str) -> Optional[str]:
        
        for edge in self.edges:
            if edge['rel'] == 'has_column' and edge['tgt'] == col_id:
                return edge['src']
        if '.' in col_id:
            return col_id.split('.')[0]
        return None
    
    def get_foreign_keys(self) -> List[Tuple[str, str]]:
        
        fks = []
        for edge in self.edges:
            if edge['rel'] == 'foreign_key_to':
                fks.append((edge['src'], edge['tgt']))
        return fks
    
    def get_primary_keys(self) -> Set[str]:
        
        pks = set()
        for edge in self.edges:
            if edge['rel'] == 'primary_key_of':
                pks.add(edge['src'])
        return pks
    
    def column_exists_in_schema(self, col_name: str) -> bool:
        col_name_norm = normalize_name(col_name)
        for col_id in self.columns:

            if normalize_name(col_id.split('.')[-1]) == col_name_norm:
                return True
        
            col_info = self.columns[col_id]
            if normalize_name(col_info.get('label', '')) == col_name_norm:
                return True
        return False
    
    def table_exists_in_schema(self, table_name: str) -> bool:
        table_name_norm = normalize_name(table_name)
        for table_id in self.tables:
            if normalize_name(table_id) == table_name_norm:
                return True
            table_info = self.tables[table_id]
            if normalize_name(table_info.get('label', '')) == table_name_norm:
                return True
        return False
    
    def find_column_id_by_name(self, col_name: str) -> Optional[str]:
        col_name_norm = normalize_name(col_name)
        for col_id in self.columns:
            if normalize_name(col_id.split('.')[-1]) == col_name_norm:
                return col_id
            col_info = self.columns[col_id]
            if normalize_name(col_info.get('label', '')) == col_name_norm:
                return col_id
        return None
    
    def is_column_required(self, col_name: str) -> bool:
        col_id = self.find_column_id_by_name(col_name)
        if not col_id:
            return False
        return col_id in self.get_required_columns()


class BaseDetector:

    def detect(self, ast: ASTNode, qss: QSSParser, question: str, **kwargs) -> List[Dict]:
        raise NotImplementedError


class ASTErrorAnnotator:
    
    
    def __init__(self, ast_dict: Dict):
        self.ast = self._init_errors(ast_dict)
    
    def _init_errors(self, node: Dict) -> Dict:
        
        new_node = {}
        for key, value in node.items():
            if key == 'children':
                new_node['children'] = [self._init_errors(c) for c in value]
            else:
                new_node[key] = value
        new_node['errors'] = []
        return new_node
    
    def annotate(self, error_label: str, 
                 target_node_type: Optional[str] = None,
                 target_value: Optional[str] = None,
                 node_path: Optional[List[int]] = None) -> bool:
        
        
        if node_path is not None:
            node = self._get_node_by_path(node_path)
            if node:
                if error_label not in node['errors']:
                    node['errors'].append(error_label)
                return True
        
        
        if target_node_type:
            nodes = self._find_nodes(self.ast, target_node_type, target_value)
            if nodes:
                for node in nodes:
                    if error_label not in node['errors']:
                        node['errors'].append(error_label)
                return True
        
        if self.ast.get('node_type') == 'select':
            if error_label not in self.ast['errors']:
                self.ast['errors'].append(error_label)
            return True
        
        return False
    
    def _get_node_by_path(self, path: List[int]) -> Optional[Dict]:
        node = self.ast
        for idx in path:
            children = node.get('children', [])
            if idx < len(children):
                node = children[idx]
            else:
                return None
        return node
    
    def _find_nodes(self, node: Dict, target_type: str, target_value: Optional[str]) -> List[Dict]:
        
        results = []
        if node.get('node_type') == target_type:
            if target_value is None:
                results.append(node)
            elif self._match_value(node, target_value):
                results.append(node)
        for child in node.get('children', []):
            results.extend(self._find_nodes(child, target_type, target_value))
        return results
    
    def _match_value(self, node: Dict, target_value: str) -> bool:
        
        node_name = normalize_name(node.get('name', ''))
        node_value = normalize_name(str(node.get('value', '')))
        target = normalize_name(target_value)
        return node_name == target or node_value == target
    
    def get_annotated_ast(self) -> Dict:
        
        return self.ast

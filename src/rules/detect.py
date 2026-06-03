import os
import sys
import json
import sqlite3
import argparse
from typing import Dict, List, Tuple, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules.base import ASTNode, QSSParser, ASTErrorAnnotator, ErrorLabels, normalize_name
from rules.attribute_error import AttributeErrorDetector
from rules.table_error import TableErrorDetector
from rules.condition_error import ConditionErrorDetector
from rules.value_error import ValueErrorDetector
from rules.function_error import FunctionErrorDetector
from rules.clause_error import ClauseErrorDetector
from rules.modifier_error import ModifierErrorDetector


def get_db_schema(db_path: str) -> Tuple[Set[str], Set[str]]:
    
    tables = set()
    columns = set()
    
    try:
        conn = sqlite3.connect(db_path, timeout=1.0)
        cursor = conn.cursor()
        
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for row in cursor.fetchall():
            table_name = row[0]
            tables.add(normalize_name(table_name))
            
            
            try:
                cursor.execute(f'PRAGMA table_info("{table_name}")')
                for col_row in cursor.fetchall():
                    columns.add(normalize_name(col_row[1]))
            except:
                pass
        
        conn.close()
    except Exception as e:
        pass
    
    return tables, columns


class HighPrecisionDetector:

    STRICT_ALIASES = {
        'rank', 'count', 'total', 'sum', 'avg', 'max', 'min', 'cnt', 'num', 
        'rownum', 'rn', 'row_number', 'dense_rank', 'year', 'month', 'day',
        'percentage', 'ratio', 'diff', 'difference'
    }
    
    ALIAS_SUFFIXES = ['_count', '_total', '_sum', '_avg', '_percent', '_ratio', 
                      '_score', '_rate', '_diff', '_budget']
    
    CTE_WORDS = {'count', 'counts', 'total', 'totals', 'ranked', 'filtered', 
                 'temp', 'sub', 'info', 'list', 'data', 'result', 'cte',
                 'time', 'times', 'speed', 'speeds', 'spent', 'budget'}
    
    def detect(self, sample: Dict, db_tables: Set[str], db_columns: Set[str]) -> List[Dict]:

        ast_dict = sample.get('abstract_syntax_tree', {})
        qss_dict = sample.get('question_schema_structure', {})
        question = sample.get('question', '')
        
        ast = ASTNode.from_dict(ast_dict)
        qss = QSSParser(qss_dict)
        
        errors = []

        ast_columns = ast.get_all_column_names()
        ast_tables = ast.get_all_table_names()
        
        seen_extra_cols = set()
        for col in ast_columns:
            col_norm = normalize_name(col)
            if col_norm and col_norm not in seen_extra_cols:
                
                if col_norm in self.STRICT_ALIASES:
                    continue
                if any(col_norm.endswith(suf) for suf in self.ALIAS_SUFFIXES):
                    continue
                
                if ' ' in col or "'" in col:
                    continue
                
                if len(col_norm) <= 2:
                    continue
                
                
                if not self._column_exists(col_norm, db_columns):
                    seen_extra_cols.add(col_norm)
                    errors.append({
                        'error_label': ErrorLabels.ATTRIBUTE_REDUNDANCY,
                        'column': col,
                        'target_node_type': 'column',
                        'target_value': col
                    })
        
        
        seen_extra_tables = set()
        for table in ast_tables:
            table_norm = normalize_name(table)
            if table_norm and table_norm not in seen_extra_tables:
                
                if self._is_likely_cte(table, table_norm):
                    continue
                
                if not self._table_exists(table_norm, db_tables):
                    seen_extra_tables.add(table_norm)
                    errors.append({
                        'error_label': ErrorLabels.TABLE_REDUNDANCY,
                        'table': table,
                        'target_node_type': 'table',
                        'target_value': table
                    })
        
        return errors
    
    def _is_likely_cte(self, table_name: str, table_norm: str) -> bool:

        capitals = sum(1 for c in table_name if c.isupper())
        if capitals >= 2 and not table_name.isupper() and not table_name.islower():
            return True
        
        
        for w in self.CTE_WORDS:
            if table_norm.endswith('_' + w) or table_norm.endswith(w + 's'):
                return True
            if w + '_' in table_norm or '_' + w in table_norm:
                return True
        
        
        if '_id' in table_norm and table_norm != 'id':
            return True
        
        return False
    
    def _column_exists(self, col_norm: str, db_columns: Set[str]) -> bool:
        
        if not col_norm:
            return True
        
        
        if len(col_norm) <= 2:
            return True
        
        
        if col_norm in db_columns:
            return True
        
        
        for db_col in db_columns:
            if col_norm in db_col or db_col in col_norm:
                return True
        
        return False
    
    def _table_exists(self, table_norm: str, db_tables: Set[str]) -> bool:
        
        if not table_norm:
            return True
        
        
        if table_norm in db_tables:
            return True
        
        
        for db_table in db_tables:
            if table_norm in db_table or db_table in table_norm:
                return True
        
        return False


def execute_sql(sql: str, db_path: str, timeout: float = 30.0) -> Tuple[Optional[List], Optional[str]]:
    
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results, None
    except Exception as e:
        return None, str(e)


def compare_results(pred_results: Optional[List], gt_results: Optional[List]) -> bool:
    
    if pred_results is None or gt_results is None:
        return False
    return set(pred_results) == set(gt_results)


def load_ground_truth_sqls(gt_path: str) -> Dict[int, Tuple[str, str]]:
    
    gt_sqls = {}
    with open(gt_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if '\t' in line:
                sql, db_name = line.rsplit('\t', 1)
                gt_sqls[idx] = (sql, db_name)
            else:
                gt_sqls[idx] = (line, '')
    return gt_sqls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--gt_sql', type=str, required=True)
    parser.add_argument('--db_root', type=str, required=True)
    parser.add_argument('--timeout', type=float, default=30.0)
    args = parser.parse_args()
    
    print(f"Loading input data from {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    
    print(f"Loading ground truth SQLs from {args.gt_sql}...")
    gt_sqls = load_ground_truth_sqls(args.gt_sql)
    print(f"Loaded {len(gt_sqls)} ground truth SQLs")
    
    detector = HighPrecisionDetector()
    
    individual_detectors = [
        AttributeErrorDetector(),
        TableErrorDetector(),
        ConditionErrorDetector(),
        ValueErrorDetector(),
        FunctionErrorDetector(),
        ClauseErrorDetector(),
        ModifierErrorDetector(),
    ]
    
    db_schema_cache = {}
    
    results = []
    
    print("Running detection...")
    for idx, sample in enumerate(data):
        question_id = sample.get('question_id', idx)
        db_id = sample.get('db_id', '')
        predicted_sql = sample.get('predicted_sql', '')
        
        gt_sql, gt_db = gt_sqls.get(question_id, ('', ''))
        
        db_path = os.path.join(args.db_root, db_id, f"{db_id}.sqlite")
        
        if db_id not in db_schema_cache:
            db_schema_cache[db_id] = get_db_schema(db_path)
        db_tables, db_columns = db_schema_cache[db_id]
        
        pred_results, pred_error = execute_sql(predicted_sql, db_path, args.timeout)
        
        gt_results, gt_error = execute_sql(gt_sql, db_path, args.timeout)
        

        exec_correct = compare_results(pred_results, gt_results)
        

        errors = detector.detect(sample, db_tables, db_columns)
        

        if pred_error:
            ast_dict = sample.get('abstract_syntax_tree', {})
            qss_dict = sample.get('question_schema_structure', {})
            question = sample.get('question', '')
            ast = ASTNode.from_dict(ast_dict)
            qss = QSSParser(qss_dict)
            
            existing_labels = {e['error_label'] for e in errors}
            for ind_detector in individual_detectors:
                exec_errors = ind_detector.detect(ast, qss, question, exec_error=pred_error)
                for exec_err in exec_errors:
                    if exec_err['error_label'] not in existing_labels:
                        errors.append(exec_err)
                        existing_labels.add(exec_err['error_label'])
        
        ast_dict = sample.get('abstract_syntax_tree', {})
        annotator = ASTErrorAnnotator(ast_dict)
        for error in errors:
            annotator.annotate(
                error['error_label'],
                target_node_type=error.get('target_node_type'),
                target_value=error.get('target_value')
            )
        
        result_entry = {
            'question_id': question_id,
            'db_id': db_id,
            'question': sample.get('question', ''),
            'question_schema_structure': sample.get('question_schema_structure', {}),
            'abstract_syntax_tree': annotator.get_annotated_ast(),
            'predicted_sql': predicted_sql,
            'exec_results': sample.get('exec_results', []),
            'external_knowledge': sample.get('external_knowledge', ''),
            'detected_errors': [e['error_label'] for e in errors],
            'exec_correct': exec_correct,
        }
        
        if pred_error:
            result_entry['exec_error'] = pred_error
        
        if errors:
            result_entry['error_label'] = errors[0]['error_label']
        
        results.append(result_entry)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(data)} samples...")
    
    print(f"Saving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Done. {len(results)} samples processed.")


if __name__ == '__main__':
    main()

import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm
import math

pbar = None

def load_json(dir):
    with open(dir, 'r') as j:
        contents = json.loads(j.read())
    return contents

def result_callback(result):
    
    global pbar
    try:
        if pbar is not None:
            try:
                pbar.update(1)
            except Exception:
                pass
    finally:
        
        exec_result.append(result)


def execute_sql(predicted_sql,ground_truth, db_path):
    
    conn = sqlite3.connect(db_path, timeout=1.0)
    try:
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        res = 0
        if set(predicted_res) == set(ground_truth_res):
            res = 1
        return res
    finally:
        try:
            conn.close()
        except Exception:
            pass

def execute_model(predicted_sql,ground_truth, db_place, idx, meta_time_out):
    try:
        res = func_timeout(meta_time_out, execute_sql,
                                  args=(predicted_sql, ground_truth, db_place))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f'timeout',)]
        res = 0
    except Exception as e:
        result = [(f'error',)]  
        res = 0

    result = {'sql_idx': idx, 'res': res}
    
    return result


def package_sqls(sql_path, db_root_path, mode='gpt', data_mode='dev', benchmark=None):
    clean_sqls = []
    db_path_list = []
    if mode == 'gpt':
        if sql_path.endswith('.json'):
            sql_data = json.load(open(sql_path, 'r'))
        else:
            sql_data = json.load(open(sql_path + 'predict_' + data_mode + '.json', 'r'))
        for idx, sql_str in sql_data.items():
            if type(sql_str) == str:
                sql, db_name = sql_str.split('\t----- ' + str(benchmark) + ' -----\t')
            else:
                sql, db_name = " ", "financial"
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    elif mode == 'gt':
        sqls = open(sql_path + data_mode + '_gold.sql')
        sql_txt = sqls.readlines()
        
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split('\t')
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    return clean_sqls, db_path_list

def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0):
    
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=num_cpus, maxtasksperchild=1)
    for i,sql_pair in enumerate(sqls):
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(
            execute_model,
            args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out),
            callback=result_callback,
        )
    pool.close()
    pool.join()

def sort_results(list_of_dicts):
  return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_acc_by_diff(exec_results, diff_json_path, benchmark):
    num_queries = len(exec_results)
    results = [res['res'] for res in exec_results]
    contents = load_json(diff_json_path)

    
    has_difficulty = any(isinstance(item, dict) and ('difficulty' in item) for item in contents)

    if not has_difficulty:
        
        all_acc = (sum(results) / num_queries) if num_queries else 0.0
        if benchmark.lower() == 'spider':
            count_lists = [0, 0, 0, 0, num_queries]
            return float('nan'), float('nan'), float('nan'), float('nan'), all_acc * 100, count_lists
        else:
            count_lists = [0, 0, 0, num_queries]
            return float('nan'), float('nan'), float('nan'), all_acc * 100, count_lists

    if benchmark.lower() == 'spider':
        
        easy_results, medium_results, hard_results, extra_results = [], [], [], []

        for i, content in enumerate(contents):
            if i >= len(exec_results):
                break
            difficulty_value = content.get('difficulty', None)
            if difficulty_value == 'easy':
                easy_results.append(exec_results[i])
            elif difficulty_value == 'medium':
                medium_results.append(exec_results[i])
            elif difficulty_value == 'hard':
                hard_results.append(exec_results[i])
            elif difficulty_value == 'extra':
                extra_results.append(exec_results[i])

        def safe_acc(bucket):
            return (sum([res['res'] for res in bucket]) / len(bucket)) if bucket else float('nan')

        easy_acc = safe_acc(easy_results)
        medium_acc = safe_acc(medium_results)
        hard_acc = safe_acc(hard_results)
        extra_acc = safe_acc(extra_results)
        all_acc = (sum(results) / num_queries) if num_queries else 0.0
        count_lists = [len(easy_results), len(medium_results), len(hard_results), len(extra_results), num_queries]
        return easy_acc * 100, medium_acc * 100, hard_acc * 100, extra_acc * 100, all_acc * 100, count_lists
    
    else: 
        simple_results, moderate_results, challenging_results = [], [], []

        for i, content in enumerate(contents):
            if i >= len(exec_results):
                break
            difficulty_value = content.get('difficulty', None)
            if difficulty_value == 'simple':
                simple_results.append(exec_results[i])
            elif difficulty_value == 'moderate':
                moderate_results.append(exec_results[i])
            elif difficulty_value == 'challenging':
                challenging_results.append(exec_results[i])

        def safe_acc(bucket):
            return (sum([res['res'] for res in bucket]) / len(bucket)) if bucket else float('nan')

        simple_acc = safe_acc(simple_results)
        moderate_acc = safe_acc(moderate_results)
        challenging_acc = safe_acc(challenging_results)
        all_acc = (sum(results) / num_queries) if num_queries else 0.0
        count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
        return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100, count_lists


def print_data(score_lists, count_lists, benchmark):
    if benchmark.lower() == 'spider':
        levels = ['easy', 'medium', 'hard', 'extra', 'total']
        
        try:
            is_train_like = (sum(count_lists[:4]) == 0) or all(math.isnan(x) for x in score_lists[:4])
        except Exception:
            is_train_like = False
    else:
        levels = ['simple', 'moderate', 'challenging', 'total']
        
        try:
            is_train_like = (sum(count_lists[:3]) == 0) or all(math.isnan(x) for x in score_lists[:3])
        except Exception:
            is_train_like = False

    if is_train_like:
        print("{:20} {:20}".format("", 'total'))
        print("{:20} {:<20}".format('count', count_lists[-1]))
        print('======================================    ACCURACY    =====================================')
        print("{:20} {:<20.2f}".format('accuracy', score_lists[-1]))
        return

    if benchmark.lower() == 'spider':
        print("{:20} {:20} {:20} {:20} {:20} {:20}".format("", *levels))
        print("{:20} {:<20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))
        print('================================================    ACCURACY    ===============================================')
        print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy', *score_lists))
    else:
        print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
        print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))
        print('======================================    ACCURACY    =====================================')
        print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy', *score_lists))


if __name__ == '__main__':
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        
        pass
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--predicted_sql_path', type=str, required=True, default='')
    args_parser.add_argument('--ground_truth_path', type=str, required=True, default='')
    args_parser.add_argument('--data_mode', type=str, required=True, default='dev')
    args_parser.add_argument('--db_root_path', type=str, required=True, default='')
    args_parser.add_argument('--num_cpus', type=int, default=1)
    args_parser.add_argument('--meta_time_out', type=float, default=30.0)
    args_parser.add_argument('--mode_gt', type=str, default='gt')
    args_parser.add_argument('--mode_predict', type=str, default='gpt')
    args_parser.add_argument('--difficulty',type=str,default='simple')
    args_parser.add_argument('--diff_json_path',type=str,default='')
    args_parser.add_argument('--benchmark', type=str, required=True)
    args = args_parser.parse_args()
    exec_result = []

    pred_queries, db_paths = package_sqls(args.predicted_sql_path, args.db_root_path, mode=args.mode_predict,
                                          data_mode=args.data_mode, benchmark=args.benchmark)
    
    gt_queries, db_paths_gt = package_sqls(args.ground_truth_path, args.db_root_path, mode='gt',
                                           data_mode=args.data_mode , benchmark=args.benchmark)

    query_pairs = list(zip(pred_queries,gt_queries))

    
    globals()['pbar'] = tqdm(total=len(query_pairs), desc='Executing SQL', dynamic_ncols=True)

    run_sqls_parallel(query_pairs, db_places=db_paths, num_cpus=args.num_cpus, meta_time_out=args.meta_time_out)

    
    try:
        _p = globals().get('pbar')
        if _p is not None:
            _p.close()
    except Exception:
        pass

    exec_result = sort_results(exec_result)
    
    
    print('Execution finished, start calculating accuracy')
    
    if args.benchmark.lower() == 'spider':
        easy_acc, medium_acc, hard_acc, extra_acc, acc, count_lists = \
            compute_acc_by_diff(exec_result, args.diff_json_path, args.benchmark)
        score_lists = [easy_acc, medium_acc, hard_acc, extra_acc, acc]
    else:
        simple_acc, moderate_acc, challenging_acc, acc, count_lists = \
            compute_acc_by_diff(exec_result, args.diff_json_path, args.benchmark)
        score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
    
    print_data(score_lists, count_lists, args.benchmark)
    if args.benchmark.lower() == 'spider':
        print('===============================================================================================================')
    else:
        print('===========================================================================================')
    print("Finished evaluation")

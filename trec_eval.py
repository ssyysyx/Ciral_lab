import pandas as pd
import tempfile
import os
import copy
from typing import Dict, Tuple
import pytrec_eval

def eval_mrr(qrels: Dict[str, Dict[str, int]],
                  results: Dict[str, Dict[str, float]], k: int = 100) -> float:
    """
    计算 MRR@k (Mean Reciprocal Rank @ k)
    
    参数:
    qrels (Dict[str, Dict[str, int]]): 查询的相关性标注，其中：
        - 键是查询ID (query_id)，值是另一个字典。
        - 内层字典的键是文档ID (doc_id)，值是相关性标注 (1 表示相关，0 表示不相关)。
    
    results (Dict[str, Dict[str, float]]): 系统返回的检索结果，其中：
        - 键是查询ID (query_id)，值是另一个字典。
        - 内层字典的键是文档ID (doc_id)，值是文档得分（用于排序）。
    
    k (int): 限制前 k 个检索结果进行计算，默认是前 100 个。
    
    返回:
    float: 计算出的 MRR@k 值。
    """
    total_reciprocal_rank = 0.0
    query_count = 0
    
    # 遍历每个查询
    for query_id, ranked_docs in results.items():
        if query_id not in qrels:
            continue  # 如果查询没有对应的qrels，跳过该查询

        # 对返回的文档按分数排序，分数高的排在前面，并限制到前 k 个文档
        sorted_docs = sorted(ranked_docs.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # 找到第一个相关文档的排名
        for rank, (doc_id, _) in enumerate(sorted_docs, start=1):
            if qrels[query_id].get(doc_id, 0) > 0:  # 如果文档是相关的
                total_reciprocal_rank += 1.0 / rank
                break
        
        query_count += 1
    
    # 计算 MRR@k，防止查询数量为0的情况
    return total_reciprocal_rank / query_count if query_count > 0 else 0.0

def trec_eval(qrels: Dict[str, Dict[str, int]],
              results: Dict[str, Dict[str, float]],
              k_values: Tuple[int] = (10, 50, 100, 200, 1000)) -> Dict[str, float]:
    ndcg, _map, recall = {}, {}, {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string})
    scores = evaluator.evaluate(results)

    for query_id in scores:
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]

    def _normalize(m: dict) -> dict:
        return {k: round(v / len(scores), 5) for k, v in m.items()}

    ndcg = _normalize(ndcg)
    _map = _normalize(_map)
    recall = _normalize(recall)

    all_metrics = {}
    for mt in [ndcg, _map, recall]:
        all_metrics.update(mt)

    return all_metrics


def get_qrels_file(name):
    THE_TOPICS = {
        'zho':'final_zho',
        'fas':'final_fas',
        'rus':'final_rus',
        'ha-dev': 'ciral-v1.0-ha-dev',
        'so-dev': 'ciral-v1.0-so-dev',
        'sw-dev': 'ciral-v1.0-sw-dev',
        'yo-dev': 'ciral-v1.0-yo-dev',
        'ha-test-a': 'ciral-v1.0-ha-test-a',
        'so-test-a': 'ciral-v1.0-so-test-a',
        'sw-test-a': 'ciral-v1.0-sw-test-a',
        'yo-test-a': 'ciral-v1.0-yo-test-a',
    }
    name = THE_TOPICS.get(name, '')
    name = 'data/label_file/qrels.' + name + '.txt'  # try to use cache
    if not os.path.exists(name):
        from pyserini.search import get_qrels_file
        return get_qrels_file(name)  # download from pyserini
    return name


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
        else:
            print('duplicate')
    return new_response


def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            try:
                new_response += str(int(c))
            except:
                new_response += ' '
    new_response = new_response.strip()
    return new_response


class EvalFunction:
    @staticmethod
    def receive_responses(rank_results, responses, cut_start=0, cut_end=100):
        print('receive_responses', len(responses), len(rank_results))
        for i in range(len(responses)):
            response = responses[i]
            response = clean_response(response)
            response = [int(x) - 1 for x in response.split()]
            response = remove_duplicate(response)
            cut_range = copy.deepcopy(rank_results[i]['hits'][cut_start: cut_end])
            original_rank = [tt for tt in range(len(cut_range))]
            response = [ss for ss in response if ss in original_rank]
            response = response + [tt for tt in original_rank if tt not in response]
            for j, x in enumerate(response):
                rank_results[i]['hits'][j + cut_start] = {
                    'content': cut_range[x]['content'], 'qid': cut_range[x]['qid'], 'docid': cut_range[x]['docid'],
                    'rank': cut_range[j]['rank'], 'score': cut_range[j]['score']}
        return rank_results

    @staticmethod
    def write_file(rank_results, file):
        print('write_file')
        with open(file, 'w') as f:
            for i in range(len(rank_results)):
                rank = 1
                hits = rank_results[i]['hits']
                for hit in hits:
                    f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                    rank += 1
        return True

    @staticmethod
    def trunc(qrels, run):
        qrels = get_qrels_file(qrels)
        # print(qrels)
        run = pd.read_csv(run, delim_whitespace=True, header=None)
        qrels = pd.read_csv(qrels, delim_whitespace=True, header=None)
        run[0] = run[0].astype(str)
        qrels[0] = qrels[0].astype(str)

        qrels = qrels[qrels[0].isin(run[0])]
        temp_file = tempfile.NamedTemporaryFile(delete=False).name
        qrels.to_csv(temp_file, sep='\t', header=None, index=None)
        return temp_file

    @staticmethod
    def main(args_qrel, args_run):
        # print(args_qrel)
        args_qrel = EvalFunction.trunc(args_qrel, args_run)

        assert os.path.exists(args_qrel)
        assert os.path.exists(args_run)

        with open(args_qrel, 'r') as f_qrel:
            qrel = pytrec_eval.parse_qrel(f_qrel)

        with open(args_run, 'r') as f_run:
            run = pytrec_eval.parse_run(f_run)

        all_metrics = trec_eval(qrel, run, k_values=(1, 5, 10, 20))
        mrr_100 = eval_mrr(qrel, run)
        print(all_metrics)
        print("mrr@100:", end = "")
        print(mrr_100)
        return all_metrics


if __name__ == '__main__':
    EvalFunction.main('dl19', 'ranking_results_file')

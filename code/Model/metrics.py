import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error


def calc_recall(rank, ground_truth, k):
    """
    calculate recall of one example
    """
    return len(set(rank[:k]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(hit, k):
    """
    calculate Precision@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)[:k]
    return np.mean(hit)


def precision_at_k_batch(hits, k):
    """
    calculate Precision@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = hits[:, :k].mean(axis=1)
    return res


def average_precision(hit, cut):
    """
    calculate average precision (area under PR curve)
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)
    precisions = [precision_at_k(hit, k + 1) for k in range(cut) if len(hit) >= k]
    if not precisions:
        return 0.
    return np.sum(precisions) / float(min(cut, np.sum(hit)))


def dcg_at_k(rel, k):
    """
    calculate discounted cumulative gain (dcg)
    rel: list, element is positive real values, can be binary
    """
    rel = np.asfarray(rel)[:k]
    dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    return dcg


def ndcg_at_k(rel, k):
    """
    calculate normalized discounted cumulative gain (ndcg)
    rel: list, element is positive real values, can be binary
    """
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(rel, k) / idcg


def ndcg_at_k_batch(hits, k):
    """
    calculate NDCG@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    hits_k = hits[:, :k]
    dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)
    idcg[idcg == 0] = np.inf

    res = (dcg / idcg)
    return res


def recall_at_k(hit, k, all_pos_num):
    """
    calculate Recall@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asfarray(hit)[:k]
    return np.sum(hit) / all_pos_num


def recall_at_k_batch(hits, k):
    """
    calculate Recall@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = (hits[:, :k].sum(axis=1) / (hits.sum(axis=1) + 0.1))
    return res


def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.


def calc_auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res


def logloss(ground_truth, prediction):
    logloss = log_loss(np.asarray(ground_truth), np.asarray(prediction))
    return logloss

def evaluate(user, item, emb_user, emb_item, item_gt, K):
    gt = np.array([item_gt[user_][-1,:] for user_ in user])
    scores = torch.tensor(emb_user @ emb_item.T)
    _, rank_indices = torch.sort(scores, descending=True)
    rank_indices = rank_indices.cpu()
    binary_hit = []
    for i in range(len(user)):
        binary_hit.append(gt[i][rank_indices[i]])
    binary_hit = np.array(binary_hit, dtype=np.float32)
    precision = precision_at_k_batch(binary_hit, K)
    recall = recall_at_k_batch(binary_hit, K)
    ndcg = ndcg_at_k_batch(binary_hit, K)
    total = sum([1 if (sum(gt_) > 0) else 0 for gt_ in gt])
    return {"ndcg": sum(ndcg)/total, "recall": sum(recall)/total, "precision": sum(precision)/total}

def calc_metrics_at_k(embedding, data, K):
    """
    cf_scores: (n_eval_users, n_eval_items)
    """
    user, query = data['user'], data['query']
    emb_user, emb_query = embedding[user], embedding[query]
    query_gt = data['query_gt']
    query_metrics = evaluate(user, query, emb_user, emb_query, query_gt, K)
    print("query result:", query_metrics)
    return query_metrics
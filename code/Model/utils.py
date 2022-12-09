from __future__ import print_function

from typing import Union, List
from collections import deque, defaultdict
from copy import deepcopy
import numpy as np
import scipy.sparse as sp
import torch
from dataclasses import dataclass, field, fields, InitVar

import scipy.sparse as sp
import yaml
from itertools import chain

# ================= #
# LOAD FILE IO UTILS #
# ================= #

def load_adj(file_adj):
    # file_adj stores a list of [{"num_nodes", "rows", "cols", "weights" (optional)} for i in range(time_step)]
    # return: adj_read: a list of sp.csr_matrix

    adj_d = np.load(file_adj, allow_pickle=True)
    adj_read = []
    for item in adj_d:
        num_nodes = item["num_nodes"]
        indptr = item['indptr']
        indices = item['indices']
        weights = item['weights']
        assert num_nodes == len(indptr) - 1
        adj = sp.csr_matrix((weights, indices, indptr), shape=[num_nodes, num_nodes])
        adj_read.append(adj)
    # return adj_read, num_nodes
    return adj_read


def load_feat(file_feat):
    feat_d = np.load(file_feat)
    return feat_d


def load_node(file_node):
    node_d = np.load(file_node)
    return node_d

def load_gt(file_gt):
    gt = np.load(file_gt, allow_pickle = True)
    # gt = list(gt_[10:20])
    # gt.append(chain(*gt_[-8:]))
    # gt = np.array(gt)
    output_gt = {}
    for time_step in range(len(gt)):
        for item in gt[time_step]:
            user, asin = item[0], item[1]
            if user not in output_gt:
                output_gt[user] = [[] for i in range(len(gt))]
            output_gt[user][time_step].append(asin)
    return output_gt, len(gt)

def prepare_gt(file_gt, pool, users, pool2id):
    pos_gt, time_step = load_gt(file_gt)
    gt = {item:np.zeros((time_step, len(pool))) for item in users}
    for user in pos_gt:
        for i in range(time_step):
            for item in pos_gt[user][i]:
                if item not in pool2id:
                    continue
                id = pool2id[item]
                gt[user][i][id] = 1
    return gt

def load_kg(file_kg):
    kg_raw = np.load(file_kg)
    total_entity = np.max(kg_raw) + 1
    entities = set(list(range(total_entity)))
    kg_pair = dict()
    kg = []
    for item in kg_raw:
        if item[0] not in kg_pair:
            kg_pair[item[0]] = [set(), set()]
        kg_pair[item[0]][0].add(item[1])
    for item in kg_pair:
        pos = kg_pair[item][0]
        kg_pair[item][0] = list(kg_pair[item][0])
        kg_pair[item][1] = np.random.choice(list(entities - pos), len(pos))
        for pos_, neg_ in zip(kg_pair[item][0], kg_pair[item][1]):
            kg.append([item, pos_, neg_])
    return np.array(kg)

def load_data(dataDir):
    files = [dataDir + item for item in ["/adjs.npy", "/feats.npy", "/nodes.npy",
                "/user.npy", "/query.npy", "/asin.npy", "/query_gt.npy", "/asin_gt.npy", "/kg.npy"]]
    [file_adj, file_feat, file_node, file_user, file_query, file_asin, file_query_gt, file_asin_gt, file_kg] = files
    adj = load_adj(file_adj)
    feat = load_feat(file_feat)
    node = load_node(file_node)
    user = load_node(file_user)
    user2id = {item:i for i, item in enumerate(user)}
    query = load_node(file_query)
    query2id = {item: i for i, item in enumerate(query)}
    asin = load_node(file_asin)
    asin2id = {item: i for i, item in enumerate(asin)}
    query_gt = prepare_gt(file_query_gt, query, user, query2id)
    asin_gt = prepare_gt(file_asin_gt, asin, user, asin2id)
    kg = load_kg(file_kg)
    bin_adj_files = [None for i in range(len(adj))]
    return {"adj": adj, "feat": feat, "node": node, "bin": bin_adj_files, "user": user,
            "query": query, "asin": asin, "query_gt": query_gt, "asin_gt": asin_gt,
            "user2id":user2id, "query2id":query2id, "asin2id":asin2id, "kg":kg}

def load_config(file_config):
    with open(file_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    sampler = []
    for s in config['sampler']:
        phase = s.pop('phase')
        sampler.append(s)
    batch_size = config["hyperparameter"]["batch_size"]
    sampler_config_ensemble = {"batch_size": batch_size, "configs": sampler}
    return sampler_config_ensemble

# ================= #
# Process ADJ UTILS #
# ================= #

def get_deg_torch_sparse(adj):
    return scatter(adj._values(), adj._indices()[0], reduce="sum")


def adj_norm_rw(adj, deg=None, dropedge=0., sort_indices=True):
    """
    Normalize adj according to the method of rw normalization.
    Note that sym norm is used in the original GCN paper (kipf),
    while rw norm is used in GraphSAGE and some other variants.
    
    # Procedure:
    #       1. adj add self-connection --> adj'
    #       2. D' deg matrix from adj'
    #       3. norm by D^{-1} x adj'
    if sort_indices is True, we re-sort the indices of the returned adj
    Note that after 'dot' the indices of a node would be in descending order
    rather than ascending order
    """
    if type(adj) == torch.Tensor:
        assert deg is None
        assert torch.sum(adj._values()).cpu().long().item() == adj._values().size()[0]
        _deg_orig = get_deg_torch_sparse(adj)
        if dropedge > 0:
            masked_indices = torch.floor(torch.rand(int(adj._values().size()[0] * dropedge)) * adj._values().size()[0]).long()
            adj._values()[masked_indices] = 0
            _deg_dropped = get_deg_torch_sparse(adj)
        else:
            _deg_dropped = _deg_orig
        _deg = torch.repeat_interleave(_deg_dropped, _deg_orig.long())
        _deg = torch.clamp(_deg, min=1)
        _val = adj._values()
        _val /= _deg
        adj_norm = adj
    else:
        assert dropedge == 0., "not supporting dropedge for scipy csr matrices"
        assert adj.shape[0] == adj.shape[1]
        diag_shape = (adj.shape[0], adj.shape[1])
        D = adj.sum(1).flatten() if deg is None else deg
        D = np.clip(D, 1, None)     # if deg_v == 0, it doesn't matter what value we clip it to. 
        norm_diag = sp.dia_matrix((1 / D, 0), shape=diag_shape)
        adj_norm = norm_diag.dot(adj)
        if sort_indices:
            adj_norm.sort_indices()
    return adj_norm


def adj_norm_sym(adj, sort_indices=True, add_self_edge=False, dropedge=0.):
    assert adj.shape[0] == adj.shape[1]
    assert adj.data.sum() == adj.size, "symmetric normalization only supports binary input adj"
    N = adj.shape[0]
    # drop edges symmetrically
    if dropedge > 0:
        masked_indices = np.random.choice(adj.size, int(adj.size * dropedge))
        adj.data[masked_indices] = 0
        adjT = adj.tocsc()
        data_add = adj.data + adjT.data
        survived_indices = np.where(data_add == 2)[0]
        adj.data *= 0
        adj.data[survived_indices] = 1
    # augment adj with self-connection
    if add_self_edge:
        indptr_new = np.zeros(N + 1)
        neigh_list = [set(adj.indices[adj.indptr[v] : adj.indptr[v+1]]) for v in range(N)]
        for i in range(len(neigh_list)):
            neigh_list[i].add(i)
            neigh_list[i] = np.sort(np.fromiter(neigh_list[i], int, len(neigh_list[i])))
            indptr_new[i + 1] = neigh_list[i].size
        indptr_new = indptr_new.cumsum()
        indices_new = np.concatenate(neigh_list)
        data_new = np.broadcast_to(np.ones(1), indices_new.size)
        adj_aug = sp.csr_matrix((data_new, indices_new, indptr_new), shape=adj.shape)
        # NOTE: no need to explicitly convert dtype, since adj_norm_sym is used for subg only
    else:
        adj_aug = adj
    # normalize
    D = np.clip(adj_aug.sum(1).flatten(), 1, None)
    norm_diag = sp.dia_matrix((np.power(D, -0.5), 0), shape=adj_aug.shape)
    adj_norm = norm_diag.dot(adj_aug).dot(norm_diag)
    if sort_indices:
        adj_norm.sort_indices()
    return adj_norm


def coo_scipy2torch(adj):
    """
    convert a scipy sparse COO matrix to torch
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(i, v, torch.Size(adj.shape))

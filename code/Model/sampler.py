from __future__ import print_function

from typing import Union, List
from collections import deque
from copy import deepcopy
import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
from dataclasses import dataclass, field, fields, InitVar
import json
import scipy.sparse as sp
import yaml

import Model.para_samplers.cpp_graph_samplers as gsp
from Model.para_samplers.base_graph_samplers import Subgraph
from Model.utils import *
from Model import TRAIN, VALID, TEST


@dataclass
class PoolSubg:
    """
    Collection of sampled subgraphs returned from the C++ sampler.
    """
    data: List[deque] = None
    # book-keeper
    num_subg: int = 0  # num subgraphs should be identical across different branches of ensemble
    num_ensemble: int = 0

    def __post_init__(self):
        self.data = [deque() for i in range(self.num_ensemble)]
        self.num_subg = 0

    def add(self, i_ens: int, subgs: list):
        for s in subgs:
            assert type(s) == Subgraph
            self.data[i_ens].append(s)
        assert self.num_subg <= len(self.data[i_ens])
        self.num_subg = len(self.data[i_ens])

    def collate(self, batch_size):
        """
        Concatenate batch_size number of subgraphs in the pool, into a single adj matrix (block diagonal form)
        e.g., ensemble = 1, batch_size = 3, and the 3 subgraphs are of size 2, 1, 3. Then the output will be:
        * subg_cat:
            x x 0 0 0 0
            x x 0 0 0 0
            0 0 x 0 0 0
            0 0 0 x x x
            0 0 0 x x x
            0 0 0 x x x
        * size_cat:
            [2, 1, 3]
        """
        subg_cat, size_cat = [], []
        for i in range(self.num_ensemble):
            subg_to_cat = []
            for _ in range(batch_size):
                subg_to_cat.append(self.data[i].popleft())
            subg_cat.append(Subgraph.cat_to_block_diagonal(subg_to_cat))
            size_cat.append([sc.indptr.size - 1 for sc in subg_to_cat])
            assert sum(size_cat[-1]) == subg_cat[-1].indptr.size - 1
        self.num_subg -= batch_size
        return subg_cat, size_cat


@dataclass
class CacheSubg:
    """
    Caching the previously sampled subgraph to be reused by later on training epochs
    """
    data: List[dict] = None
    # book-keeper
    num_recorded: List[int] = None
    # init var
    _num_ens: InitVar[int] = 0

    def __post_init__(self, _num_ens: int):
        self.data = [{} for _ in range(_num_ens)]
        self.num_recorded = [0 for _ in range(_num_ens)]

    def get(self, i_ens: int, i_subg: int):
        return self.data[i_ens][i_subg]

    def set(self, i_ens: int, i_subg: int, subg: Subgraph):
        self.data[i_ens][i_subg] = subg
        self.num_recorded[i_ens] += 1

    def is_empty(self, i_ens: int):
        return len(self.data[i_ens]) == 0


class RecSampler:

    def __init__(self, data, bin_adj_files, sampler_config_ensemble, parallelism):

        # data initial:
        adjs, feats, nodes, bin_adj_files = data["adj"], data["feat"], data["node"], data["bin"]
        self.batch_num, self.batch_size = -1, {TRAIN: 0, VALID: 0, TEST: 0}
        self.nodes = data["user"]
        self.adjs = adjs
        self.feats = feats
        self.time_step = len(adjs)
        self.sampler_config_ensemble = sampler_config_ensemble
        self.bin_adj_files = bin_adj_files
        self.query_gt = data["query_gt"]
        self.asin_gt = data["asin_gt"]
        self.user_node = data["user"]
        self.query_node = data["query"]
        self.asin_node = data["asin"]
        self.kg = data["kg"]

        # sampler initial:
        self.num_ensemble = 0
        self.parallelism = parallelism
        self.graph_sampler = [None for i in range(self.time_step)]
        sampler_config_ensemble = deepcopy(sampler_config_ensemble)
        self.num_ensemble = 0
        for sc in sampler_config_ensemble['configs']:
            num_ens_cur = [len(v) for k, v in sc.items() if k != 'method']
            if len(num_ens_cur) == 0:
                self.num_ensemble += 1
            else:
                assert max(num_ens_cur) == min(num_ens_cur)
                self.num_ensemble += num_ens_cur[0]
        if "full" in [c['method'] for c in sampler_config_ensemble['configs']]:
            # treat FULL sampler as no sampling. Also no ensemble under FULL sampler
            assert self.num_ensemble == 1
            for m in range(self.time_step):
                self.batch_size[m] = self.node_set[m].size
            self.mode_sample = self.FULL
        else:
            self.record_subgraphs = {}
            self.args_sampler_init = [sampler_config_ensemble, parallelism, bin_adj_files]
        for m in range(self.time_step):
            self.record_subgraphs[m] = ['noncache'] * self.num_ensemble
        self.cache_subg, self.pool_subg = {}, {}
        for m in range(self.time_step):
            self.cache_subg[m] = CacheSubg(_num_ens=self.num_ensemble)
            self.pool_subg[m] = PoolSubg(num_ensemble=self.num_ensemble)
        self.instantiate_sampler()
        for i in range(self.time_step):
            self.epoch_start_reset(i)

    def instantiate_sampler(self):
        sampler_config_ensemble_ = deepcopy(self.sampler_config_ensemble)
        config_ensemble = []
        # e.g., input: [{"method": "ppr", "k": [50, 10]}, {"method": "khop", "depth": [2], "budget": [10]}]
        #       output: [{"method": "ppr", "k": 50}, {"method": "ppr", "k": 10}, {"method": "khop", "depth": 2, "budget": 10}]
        for cfg in sampler_config_ensemble_["configs"]:  # different TYPEs of samplers
            method = cfg.pop('method')
            cnt_cur_sampler = [len(v) for k, v in cfg.items()]
            assert len(cnt_cur_sampler) == 0 or max(cnt_cur_sampler) == min(cnt_cur_sampler)
            cnt_cur_sampler = 1 if len(cnt_cur_sampler) == 0 else cnt_cur_sampler[0]
            cfg['method'] = [method] * cnt_cur_sampler
            cfg_decoupled = [{k: v[i] for k, v in cfg.items()} for i in range(cnt_cur_sampler)]
            config_ensemble.extend(cfg_decoupled)
        self.num_ensemble = len(config_ensemble)
        config_ensemble_mode = {}
        for index in range(self.time_step):
            self.batch_size[index] = sampler_config_ensemble_["batch_size"]
            config_ensemble_mode[index] = deepcopy(config_ensemble)
        for cfg in config_ensemble:
            assert "method" in cfg
            assert "size_root" not in cfg or cfg["size_root"] == 1
        for cfg_mode, cfg_ensemble in config_ensemble_mode.items():
            # print(cfg_mode, "hahah")
            for cfg in cfg_ensemble:
                cfg["size_root"] = 1  # we want each target to have its own subgraph
                cfg[
                    "fix_target"] = True  # i.e., we differentiate root node from the neighbor nodes (compare with GraphSAINT)
                cfg["sequential_traversal"] = True  # (mode != "train")
                if cfg["method"] == "ppr":
                    cfg["type_"] = TRAIN
                    # cfg['name_data'] = self.name_data
                    # cfg["dir_data"] = self.dir_data
                    # cfg['is_transductive'] = self.is_transductive
            aug_feat_ens = [set()] * len(cfg_ensemble)
            self.graph_sampler[cfg_mode] = gsp.GraphSamplerEnsemble(
                self.adjs[cfg_mode], self.nodes, cfg_ensemble, aug_feat_ens,
                max_num_threads=self.parallelism, num_subg_per_batch=200, bin_adj_files=self.bin_adj_files[cfg_mode]
            )

    def epoch_start_reset(self, mode):
        """
        Reset structs related with later on reuse of sampled subgraphs.
        """
        self.batch_num = -1
        if self.graph_sampler[mode] is None and self.mode_sample == self.SUBG:
            self.instantiate_sampler(*self.args_sampler_init, modes=[mode])
            if mode not in self.nocache_modes:
                self.record_subgraphs[mode] = ["record" if g.name in REUSABLE_SAMPLER else "none" for g in
                                               self.graph_sampler[mode].sampler_list]
            else:
                self.record_subgraphs[mode] = ['noncache'] * self.num_ensemble
        # elif self.mode_sample == self.FULL:
        #     return
        self.graph_sampler[mode].return_target_only = []
        for i in range(len(self.record_subgraphs[mode])):
            if self.record_subgraphs[mode][i] == "reuse":
                self.graph_sampler[mode].return_target_only.append(True)
            else:
                self.graph_sampler[mode].return_target_only.append(False)

    def par_graph_sample(self, index):
        """
        Perform graph sampling in parallel. A wrapper function for graph_samplers.py
        """
        subg_ens_l_raw = self.graph_sampler[index].par_sample_ensemble()
        for i, subg_l_raw in enumerate(subg_ens_l_raw):
            subg_ens_l = None
            if self.record_subgraphs[index][i] == "record":
                for subg in subg_l_raw:
                    assert subg.target.size == 1
                    id_root = subg.node[subg.target][0]
                    self.cache_subg[index].set(i, id_root, subg)
                subg_ens_l = subg_l_raw
            elif self.record_subgraphs[index][i] == "reuse":
                subg_ens_l = []
                for subg in subg_l_raw:
                    assert subg.node.size == 1
                    id_root = subg.node[0]
                    subg_ens_l.append(self.cache_subg[index].get(i, id_root))
            elif self.record_subgraphs[index][i] in ['noncache', 'none']:
                subg_ens_l = subg_l_raw
            else:
                raise NotImplementedError
            self.pool_subg[index].add(i, subg_ens_l)

    def load_feature(self, nodes, time_step):
        return torch.tensor(self.feats[time_step][nodes], dtype = torch.float32)

    def load_gt(self, targets, time_step):
        gt_query, gt_asin = [], []
        for node in targets:
            query_label = self.query_gt[node][time_step]
            asin_label = self.asin_gt[node][time_step]
            gt_query.append(query_label)
            gt_asin.append(asin_label)
        return [np.array(gt_query), np.array(gt_asin)]

    def rec_loader(self, batch_size, ret_raw_idx=False):
        data = []
        output_data = []
        for i in range(0, len(self.nodes), batch_size):
            adj_ens_t, feat_ens_t, target_ens_t, sampled_node_t, size_subg_ens_t, gt_batch_t = [], [], [], [], [], []
            if len(self.nodes) - i - 1 >= batch_size:
                batch_size_ = batch_size
            else:
                batch_size_ = len(self.nodes) - i - 1
            for index in range(self.time_step):
                adj_ens, feat_ens, target_ens, sampled_node = [], [], [], []
                while self.pool_subg[index].num_subg < batch_size_:
                    self.par_graph_sample(index)
                subgs_ens, size_subg_ens = self.pool_subg[index].collate(batch_size_)
                size_subg_ens = torch.tensor(size_subg_ens)
                label_idx = None
                gt_batch = None
                for subgs in subgs_ens:
                    #####
                    #print(type(subgs.target), type(subgs.node))
                    output_data.append([list(subgs.target), list(subgs.node)])
                    #####
                    sampled_node.append(subgs.node)
                    #print(index, subgs.node[0])
                    assert subgs.target.size == batch_size_
                    if label_idx is None:
                        label_idx = subgs.node[subgs.target]
                    else:
                        assert np.all(label_idx == subgs.node[subgs.target])
                    adj_ens.append(subgs.to_csr_sp())
                    feat_ens.append(torch.tensor(self.feats[index][subgs.node], dtype = torch.float32))
                    target_ens.append(torch.tensor(subgs.target))
                    if gt_batch is None:
                        gt_batch = self.load_gt(subgs.node[subgs.target], index)
                    # if 'hops' in self.aug_feats:
                    #     feat_aug_ens[-1]['hops'] = torch.tensor(self.hop2onehot_vec(subgs.hop).astype(np.float32))
                    # if 'pprs' in self.aug_feats:
                    #     feat_aug_ens[-1]['pprs'] = torch.tensor(self.ppr2onehot_vec(subgs.ppr).astype(np.float32))
                adj_ens_t.append(adj_ens)
                feat_ens_t.append(feat_ens)
                target_ens_t.append(target_ens)
                sampled_node_t.append(sampled_node)
                size_subg_ens_t.append(size_subg_ens)
                gt_batch_t.append(gt_batch)
            ret = {"adj_ens"        : adj_ens_t,
                    "feat_ens"       : feat_ens_t,
                    "size_subg_ens"  : size_subg_ens_t,
                    "target_ens"     : target_ens_t,
                    "sampled_node"   : sampled_node_t,
                    "gt_batch_t"    : gt_batch_t}
            data.append(ret)
        print(len(output_data))
        # with open("subgraph_analysis.json", "w") as f:
        #     json.dump(output_data, f)
        np.save("subgraph_analysis.npy", output_data)
        return data

    def kg_loader(self, kg_batch_size):

        data = []
        for i in range(0, len(self.kg), kg_batch_size):
            if i + kg_batch_size < len(self.kg):
                data.append(self.kg[i:i+kg_batch_size])
            else:
                data.append(self.kg[i:])
        return data


    def data_loader(self, args, ret_raw_idx=False):
        """
        Prepare one batch of training subgraph. For each root, the sampler returns the subgraph adj separatedly.
        To improve the computation efficiency, we concatenate the batch number of separate adj into a single big adj.
        i.e., the resulting adj is of the block-diagnol form.
        """
        # if self.graph_sampler[mode] is None:    # no sampling, return the full graph.
        #     self._update_batch_stat(mode, self.batch_size[mode])
        #     targets = self.node_set[mode]
        #     assert ret_raw_idx, "None subg mode should only be used in preproc!"
        #     return {"adj_ens": [self.adj[mode]], "feat_ens": [self.feat_full], "label_ens": [self.label_full],
        #             "size_subg_ens": None, "target_ens": [targets], "feat_aug_ens": None, "idx_raw": [np.arange(self.adj[mode].shape[0])]}
        rec_batch_size = args.rec_batch_size
        kg_batch_size = args.kg_batch_size
        rec_data = self.rec_loader(rec_batch_size)
        kg_data = self.kg_loader(kg_batch_size)

        return rec_data, kg_data



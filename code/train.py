from __future__ import print_function

from typing import Union, List
from collections import deque
from copy import deepcopy
import numpy as np
import datetime
import scipy.sparse as sp
import torch
import Model.para_samplers.cpp_graph_samplers as gsp
from Model.para_samplers.base_graph_samplers import Subgraph
from collections import defaultdict
from dataclasses import dataclass, field, fields, InitVar
from Model.model import Model
import scipy.sparse as sp
import yaml

from Model.utils import *
from Model.sampler import RecSampler
from Model.metrics import *
from Model.arg_parser import parse_args

args = parse_args()
device = args.device

data = load_data(args.data_dir + args.data_name)
print("finish loading data...")
sampler_config_ensemble = load_config(args.sampler_config)
print("finish loading config file...")

sampler = RecSampler(data, [None for i in range(args.time_step)], sampler_config_ensemble, 96)
rec_data, kg_data = sampler.data_loader(args)

entity_emb = torch.tensor(data["feat"][0], dtype = torch.float32)
print(data["feat"].shape)
user_emb = torch.tensor(data["feat"][:-1,data["user"]], dtype = torch.float32)


input_data = {"query": torch.tensor(data["query"], dtype = int),
              "asin": torch.tensor(data["asin"], dtype = int),
              "user": torch.tensor(data["user"], dtype = int),
              "entity_emb": entity_emb, "user_emb": user_emb}
parameter = {"time_step": args.time_step - 1, "dim_in":  args.entity_dim, "dim_out": args.entity_dim, "mulhead":1, "num_layers":2,
             "dropout":0.0, "ensemble":1, "margin": args.margin, "device":args.device}
model = Model(input_data, parameter)
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1)

print(list(model.state_dict()))

for epoch in range(1, args.n_epoch):
    model.train()
    # update KG embedding:
    kg_total_loss, rec_total_loss = [], []
    for batchID, batch in enumerate(kg_data):
        h = torch.tensor(batch[:,0], dtype = int).to(device)
        t_pos = torch.tensor(batch[:,1], dtype = int).to(device)
        t_neg = torch.tensor(batch[:,2], dtype = int).to(device)

        kg_loss = model.optimize_kg(h, t_pos, t_neg)
        optimizer.zero_grad()
        kg_loss.backward()
        optimizer.step()
        kg_total_loss.append(kg_loss.item())
        print('Epoch: {}/{}\tKG training: {}/{}\t loss: {}'.format(epoch, args.n_epoch, batchID, len(kg_data) - 1, kg_loss.item()), end="\r")
    #print()
    for batchID, batch in enumerate(rec_data):
        adj_ens_t = batch["adj_ens"]
        size_subg_ens_t = batch["size_subg_ens"]
        target_ens_t = batch["target_ens"]
        sampled_node_t = batch["sampled_node"]
        gt_batch_t = batch["gt_batch_t"]
        if epoch == 0:
            for i in range(len(target_ens_t)):
                for j in range(parameter['ensemble']):
                    target_ens_t[i][j] = target_ens_t[i][j].to(device)
                    size_subg_ens_t[i][j] = size_subg_ens_t[i][j].to(device)
                    sampled_node_t[i][j] = torch.tensor(sampled_node_t[i][j], dtype = int).to(device)
                    for i in range(len(gt_batch_t)):
                        gt_batch_t[i][0] = torch.tensor(gt_batch_t[i][0], dtype=int).to(device)
                        gt_batch_t[i][1] = torch.tensor(gt_batch_t[i][1], dtype=int).to(device)
        optimizer.zero_grad()
        output, rec_loss, _ = model.optimize_rec(adj_ens_t, target_ens_t, sampled_node_t, size_subg_ens_t, gt_batch_t)
        rec_loss.backward()
        optimizer.step()
        rec_total_loss.append(rec_loss.item())
        print('Epoch: {}/{}\tRecommendation training: {}/{}\t loss: {}'.format(epoch, args.n_epoch, batchID, len(rec_data) - 1, rec_loss.item()), end="\r")
    #print()
    print('Epoch: {}/{}\t, KG loss: {} \t Rec loss: {}\t Time: {}'.format(epoch, args.n_epoch, np.mean(kg_total_loss), np.mean(rec_total_loss), datetime.datetime.now()))
    if epoch % 10:
        continue
    model.eval()
    embedding = torch.tensor(data["feat"][0], dtype=torch.float32)
    entity_emb = model.KG_emb.entity_emb().cpu().detach()
    embedding[:entity_emb.shape[0]] = entity_emb
    for batchID, batch in enumerate(rec_data):
        adj_ens_t = batch["adj_ens"]
        size_subg_ens_t = batch["size_subg_ens"]
        target_ens_t = batch["target_ens"]
        sampled_node_t = batch["sampled_node"]
        gt_batch_t = batch["gt_batch_t"]
        optimizer.zero_grad()
        output, _, attns = model.optimize_rec(adj_ens_t, target_ens_t, sampled_node_t, size_subg_ens_t, gt_batch_t, "pred")
        userIDs = sampled_node_t[0][0][target_ens_t[0][0]]
        embedding[userIDs] = output.cpu().detach()[:,-1,:]
    _, _ = calc_metrics_at_k(embedding, data, args.K)

    #evaluation(embedding, data, time_step = args.time_step, k = 20)
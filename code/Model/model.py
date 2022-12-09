import torch
from torch import nn
import scipy.sparse as sp
import torch.nn.functional as F
from Model.utils import adj_norm_sym, adj_norm_rw, coo_scipy2torch, get_deg_torch_sparse
from torch_scatter import scatter
from torch_geometric.nn import global_sort_pool
from torch_geometric.utils import softmax
import numpy as np
from Model.layers import *
from Model.metrics import *

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

def warpLoss(scores, target, margin):
   err = torch.tensor(0.0).to(scores.device)
   for b in range(scores.size(0)):
       score_, target_ = scores[b], target[b]
       pos_index, neg_index = torch.eq(target_, 1.), torch.eq(target_, 0.)
       pos_score, neg_score = score_[pos_index], score_[neg_index]
       discrim = margin - pos_score.repeat(neg_score.size(0), 1).t() + neg_score.repeat(pos_score.size(0), 1)
       #err += F.relu(discrim).sum()
       rank = discrim.gt(0).sum(axis=1).float().unsqueeze(1)
       score = F.relu(discrim).div(rank)
       score[score != score] = 0
       err += (torch.log1p(rank) * score.sum(axis=1).unsqueeze(1)).sum()
   return err/len(scores)

class EnsembleDummy(nn.Module):
    """ used when there is only one branch of subgraph (i.e., no ensemble) """

    def __init__(self, dim_in=0, dim_out=0, **kwargs):
        super().__init__()

    def forward(self, Xi):
        assert len(Xi) == 1, "ONLY USE DUMMY ENSEMBLER WITH ONE BRANCH!"
        return Xi[0]

    def complexity(self, dims):
        assert len(dims) == 1, "ONLY USE DUMMY ENSEMBLER WITH ONE BRANCH!"
        return dims[0], 0

class EnsembleAggregator(nn.Module):
    def __init__(self, dim_in, dim_out, num_ensemble, dropout=0.0, act="leakyrelu", type_dropout="none"):
        r"""
        Embedding matrix from branches: X_i
        Output matrix after ensemble: Y

        Learnable parameters (shared across branches):
        * W \in R^{dim_in x dim_out}
        * b \in R^{dim_out}
        * q \in R^{dim_out}

        Operations:
        1. w_i = act(X_i W + b) q
        2. softmax along the i dimension
        3. Y = \sum w_i X_i
        """
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.act = nn.ModuleList(get_torch_act(act, locals()) for _ in range(num_ensemble))
        self.f_lin = nn.Linear(dim_in, dim_out, bias=True)
        self.f_dropout = nn.Dropout(p=self.dropout)
        self.q = nn.Parameter(torch.ones(dim_out))
        assert type_dropout in ["none", "feat", "coef"]
        self.type_dropout = type_dropout

    def forward(self, Xi):
        omega_ensemble = []
        for i, X in enumerate(Xi):
            if self.type_dropout == "none":
                X_ = X
            elif self.type_dropout == "coef":
                X_ = self.f_dropout(X)
            else:
                Xi[i] = self.f_dropout(X)
                X_ = Xi[i]
            omega_ensemble.append(self.act[i](self.f_lin(X_)).mm(self.q.view(-1, 1)))
        omega_ensemble = torch.cat(omega_ensemble, 1)
        omega_norm = F.softmax(omega_ensemble, dim=1)
        Y = 0
        for i, X in enumerate(Xi):
            Y += omega_norm[:, i].view(-1, 1) * X
        return Y

class KG_emb(nn.Module):
    def __init__(self, in_dim, out_dim, init_emb, l2loss = 0.005, pre_trained = "External", method = 'inner product'):
        super().__init__()

        # hyperparameter:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_entity = init_emb.shape[0]
        self.l2loss = l2loss
        self.embedding = nn.Embedding(self.n_entity, self.in_dim)
        if pre_trained == "External":
            self.embedding.weight = nn.Parameter(init_emb)
        elif pre_trained == "BERT":
            self.embedding.weight = nn.Parameter(init_emb)
        self.method = method

    def forward(self, h, pos_t, neg_t):
        h_embed = self.embedding(h)  # (kg_batch_size, entity_dim)
        pos_t_embed = self.embedding(pos_t)  # (kg_batch_size, entity_dim)
        neg_t_embed = self.embedding(neg_t)  # (kg_batch_size, entity_dim)

        if self.method == "inner product":
            pos_score = torch.einsum('bs,bs->b', h_embed, pos_t_embed)  # (kg_batch_size)
            neg_score = torch.einsum('bs,bs->b', h_embed, neg_t_embed)  # (kg_batch_size)

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)
        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
        loss = kg_loss + self.l2loss * l2_loss
        return loss

    def entity_emb(self):
        return self.embedding.weight

class Subgraph_emb(nn.Module):
    def __init__(self, time_step, in_dim, out_dim, dropout = 0.0, mulhead = 1, ensemble = 1):
        super().__init__()
        # hyperparameter:
        self.time_step = time_step
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.mulhead = mulhead
        self.ensemble = ensemble
        # module:
        self.structure1 = GCN(in_dim, out_dim, dropout = self.dropout, mulhead = self.mulhead)
        self.structure2 = GCN(in_dim, out_dim, dropout = self.dropout, mulhead = self.mulhead)
        self.structure3 = GCN(in_dim, out_dim, dropout=self.dropout, mulhead=self.mulhead)
        # self.structure4 = GCN(in_dim, out_dim, dropout = self.dropout, mulhead = self.mulhead)
        # self.structure5 = GCN(in_dim, out_dim, dropout=self.dropout, mulhead=self.mulhead)
        #self.structure = [self.structure1, self.structure2, self.structure3]
        self.structure = [self.structure1,self.structure2]
        self.subg_pooling = ResPool(out_dim, out_dim, 2, "sum", "mean",
                    dropout=self.dropout, args_pool=dict())
        if ensemble > 1:
            self.ensembler = EnsembleAggregator(dim_in = self.in_dim, dim_out = self.out_dim, num_ensembl = ensemble)
        else:
            self.ensembler = EnsembleDummy()

    def forward(self, feat_ens, adj_ens, target_ens, size_subg_ens, is_normed = False, dropedge = 0):
        output_ens = []
        for i in range(self.ensemble):
            output = []
            input = feat_ens[i]
            for j in range(len(self.structure)):
                output_ = self.structure[j]([input, adj_ens[i], is_normed, dropedge])
                #print("GCN output shape", output_[0].shape)
                input = output_[0]
                output.append(output_[0])
            emb_subg_i = self.subg_pooling(output, target_ens[i], size_subg_ens[i])
            emb_subg_i = F.normalize(emb_subg_i, p=2, dim=1)
            output_ens.append(emb_subg_i)
        emb_ensemble = self.ensembler(output_ens)
        return emb_ensemble

class Model(nn.Module):
    def __init__(self, input_data, parameter):
        super().__init__()
        # hyperparameter:
        self.dim_in = parameter["dim_in"]
        self.dim_out = parameter["dim_out"]
        self.time_step = parameter["time_step"]
        self.mulhead = parameter["mulhead"]
        self.num_layers = parameter["num_layers"]
        self.dropout = parameter["dropout"]
        self.dropedge = 0.0
        self.ensemble = parameter["ensemble"]
        self.ranking_margin = parameter["margin"]
        self.device = parameter["device"]

        # data:
        self.user = input_data["user"]
        self.query = input_data["query"]
        self.asin = input_data["asin"]

        # module:
        self.KG_emb = KG_emb(self.dim_in, self.dim_out, input_data["entity_emb"])
        self.Subgraph_emb = Subgraph_emb(self.time_step, self.dim_in, self.dim_out, dropout = self.dropout, mulhead = self.mulhead, ensemble = self.ensemble)
        self.temporal_embedding = TemporalAttentionLayer(input_dim = self.dim_out, n_heads = 1, num_time_steps = self.time_step, attn_drop = self.dropout, residual = True)
        self.l2loss = nn.MSELoss()
        self.ranking_loss_ = torch.nn.MarginRankingLoss(margin=0.5, size_average=None, reduce=None, reduction='mean')
        self.user_emb = input_data["user_emb"].to(self.device)
        self.entity_emb = self.KG_emb.entity_emb()

    def optimize_kg(self, h, pos_t, neg_t):
        loss = self.KG_emb(h, pos_t, neg_t)
        return loss

    def update_emb(self):
        self.entity_emb = self.KG_emb.entity_emb()

    def get_user_id(self, IDs):
        return
    def optimize_rec(self, adj_ens, target_ens, sampled_node, size_subg_ens, gt_batch, mode = "train"):
        output = []
        embedding = self.entity_emb
        for i in range(self.time_step):
            feat_ens = [embedding[sampled_node[i][j]] for j in range(self.ensemble)]
            output_ = self.Subgraph_emb(feat_ens, adj_ens[i], target_ens[i], size_subg_ens[i])
            output.append(output_)
        structural_outputs = [x[:, None, :] for x in output]  # list of [Ni, 1, F]
        temporal_input = torch.cat(structural_outputs, dim=1) #[N, T, F]
        output, attns = self.temporal_embedding(temporal_input)
        #print(output.shape)

        # userIDs = sampled_node[0][0][target_ens[0][0]]
        # self.user_emb[:,userIDs] = output.detach().permute(1,0,2)
        if mode == "train":
            rec_loss = self.loss(output, gt_batch)
        elif mode == "pred":
            rec_loss = None
        return output, rec_loss, attns

    def loss(self, output, gt_batch):
        device = output.device
        asin_ranking_loss = torch.tensor(0.0, dtype = torch.float32).to(device)
        query_ranking_loss = torch.tensor(0.0, dtype=torch.float32).to(device)

        query_emb = self.entity_emb[self.query]
        asin_emb = self.entity_emb[self.asin]

        for i in range(self.time_step-1):
            query_gt = torch.tensor(gt_batch[i+1][0], dtype = int).to(device)
            asin_gt = torch.tensor(gt_batch[i+1][1], dtype = int).to(device)
            query_score = nn.Sigmoid()(output[:,i,:] @ query_emb.T)
            asin_score = nn.Sigmoid()(output[:,i,:] @ asin_emb.T)
            query_ranking_loss += warpLoss(query_score, query_gt, self.ranking_margin)
            asin_ranking_loss += warpLoss(asin_score, asin_gt, self.ranking_margin)
            del query_gt
            del asin_gt
        loss_ = asin_ranking_loss + query_ranking_loss
        return loss_

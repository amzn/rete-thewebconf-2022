import torch
from torch import nn
import scipy.sparse as sp
import torch.nn.functional as F
from Model.utils import adj_norm_sym, adj_norm_rw, coo_scipy2torch, get_deg_torch_sparse
from torch_scatter import scatter
from torch_geometric.nn import global_sort_pool
from torch_geometric.utils import softmax
import numpy as np
import torch.nn.functional as F


from collections import namedtuple


import copy

F_ACT = {"relu"     : (nn.ReLU, {}),
         "I"        : (nn.LeakyReLU, {"negative_slope": (lambda kwargs: 1)}),
         "elu"      : (nn.ELU, {}),
         "tanh"     : (nn.Tanh, {}),
         "leakyrelu": (nn.LeakyReLU, {"negative_slope": (lambda kwargs: 0.2)}),
         "prelu"    : (nn.PReLU, {}),
         "prelu+"   : (nn.PReLU, {"num_parameters": (lambda kwargs: kwargs['dim_out'])})}


def get_torch_act(act, args):
    _torch_args = {k: v(args) for k, v in F_ACT[act][1].items()}
    return F_ACT[act][0](**_torch_args)


class ResPool(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_layers: int, type_res: str, type_pool: str,
                 dropout: float, args_pool: dict = None):
        super().__init__()
        self.dim_out = dim_out
        self.type_pool = type_pool
        self.type_res = type_res
        if type_pool == 'center':
            if type_res == 'none':
                self.dim_in = self.dim_out = 0
            elif type_res in ['cat', 'concat']:
                # This is equivalent to regular JK
                # 1. take center node for each feat-\ell
                # 2. concatenate multi-scale feat of center nodes
                # 3. MLP
                self.dim_in = num_layers * dim_in
            else:  # replace step 2 with max / mean
                self.dim_in = dim_in
        else:  # e.g., sort / max / mean / sum
            # 1. pool all # layer feats
            # 2. cat center node for each layer
            # 3. cat outputs of 1. and 2., feed to MLP
            if type_res in ['cat', 'concat']:
                self.dim_in = 2 * dim_in * num_layers  # MLP dimension after pooling
            else:
                self.dim_in = 2 * dim_in
            if type_pool == 'sort':
                # [pool input]        -> [pool MLP input]    -> [pool MLP output]
                # N * self.dim_in / 2 -> k * self.dim_in / 2 -> self.dim_in / 2
                assert 'k' in args_pool, "Sort pooling needs the budget k as input!"
                self.k = args_pool['k']
                _f_lin_pool = nn.Linear(self.k * int(self.dim_in / 2), int(self.dim_in / 2))
                _f_dropout_pool = nn.Dropout(p=dropout)
                _act_pool = nn.ReLU()
                self.nn_pool = nn.Sequential(_f_dropout_pool, _f_lin_pool, _act_pool)
        if self.dim_in > 0 and self.dim_out > 0:
            _act = nn.ReLU()
            _f_lin = nn.Linear(self.dim_in, self.dim_out, bias=True)
            _f_dropout = nn.Dropout(p=dropout)
            self.nn = nn.Sequential(_f_dropout, _f_lin, _act)
            self.offset = nn.Parameter(torch.zeros(self.dim_out))
            self.scale = nn.Parameter(torch.ones(self.dim_out))

    def f_norm(self, _feat):
        mean = _feat.mean(dim=1).view(_feat.shape[0], 1)
        var = _feat.var(dim=1, unbiased=False).view(_feat.shape[0], 1) + 1e-9
        feat_out = (_feat - mean) * self.scale * torch.rsqrt(var) + self.offset
        return feat_out

    def f_residue(self, feat_l):
        # 'none' residue is handled separately in forward()
        if self.type_res in ['cat', 'concat']:
            feat_ret = torch.cat(feat_l, dim=1)
        elif self.type_res == 'sum':
            feat_ret = torch.stack(feat_l, dim=0).sum(dim=0)
        elif self.type_res == 'max':
            feat_ret = torch.max(torch.stack(feat_l, dim=0), dim=0).values
        else:
            raise NotImplementedError
        return feat_ret

    def f_res_complexity(self, dim_x_l):
        """
        returns feature dim after residue, ops due to residue aggregation
        """
        if self.type_res in ['cat', 'concat']:
            return sum([d.num_feats for d in dim_x_l]), 0
        elif self.type_res == 'sum':
            return dim_x_l[-1].num_feats, (len(dim_x_l) - 1) * (dim_x_l[-1].num_nodes * dim_x_l[-1].num_feats)
        elif self.type_res == 'max':
            return dim_x_l[-1].num_feats, (len(dim_x_l) - 1) * (dim_x_l[-1].num_nodes * dim_x_l[-1].num_feats)
        else:
            raise NotImplementedError

    def forward(self, feats_in_l, idx_targets, sizes_subg):
        if self.type_pool == 'center':
            if self.type_res == 'none':
                return feats_in_l[-1][idx_targets]
            else:  # regular JK
                feats_root_l = [f[idx_targets] for f in feats_in_l]
                feat_in = self.f_residue(feats_root_l)
        elif self.type_pool in ['max', 'mean', 'sum']:
            # first pool subgraph at each layer, then residue
            offsets = torch.cumsum(sizes_subg, dim=0)
            offsets = torch.roll(offsets, 1)
            offsets[0] = 0
            offsets = offsets.to(feats_in_l[-1].device)
            idx = torch.arange(feats_in_l[-1].shape[0]).to(feats_in_l[-1].device)
            if self.type_res == 'none':
                feat_pool = F.embedding_bag(idx, feats_in_l[-1], offsets, mode=self.type_pool)
                feat_root = feats_in_l[-1][idx_targets]
            else:
                feat_pool_l = []
                for feat in feats_in_l:
                    feat_pool = F.embedding_bag(idx, feat, offsets, mode=self.type_pool)
                    feat_pool_l.append(feat_pool)
                feat_pool = self.f_residue(feat_pool_l)
                feat_root = self.f_residue([f[idx_targets] for f in feats_in_l])
            feat_in = torch.cat([feat_root, feat_pool], dim=1)
        elif self.type_pool == 'sort':
            if self.type_res == 'none':
                feat_pool_in = feats_in_l[-1]
                feat_root = feats_in_l[-1][idx_targets]
            else:
                feat_pool_in = self.f_residue(feats_in_l)
                feat_root = self.f_residue([f[idx_targets] for f in feats_in_l])
            arange = torch.arange(sizes_subg.size(0)).to(sizes_subg.device)
            idx_batch = torch.repeat_interleave(arange, sizes_subg)
            feat_pool_k = global_sort_pool(feat_pool_in, idx_batch, self.k)  # #subg x (k * F)
            feat_pool = self.nn_pool(feat_pool_k)
            feat_in = torch.cat([feat_root, feat_pool], dim=1)
        else:
            raise NotImplementedError
        return self.f_norm(self.nn(feat_in))

    def complexity(self, dims_x_l, sizes_subg):
        num_nodes = len(sizes_subg)
        num_neigh = dims_x_l[-1].num_nodes
        assert num_neigh == sizes_subg.sum()
        if self.type_pool == 'center':
            if self.type_res == 'none':
                return Dims_X(num_nodes, dims_x_l[-1].num_feats), 0
            else:  # regular JK
                dims_root_l = [Dims_X(num_nodes, d.num_feats) for d in dims_x_l]
                dim_f, ops = self.f_res_complexity(dims_root_l)  # pool first, then residue
                return Dims_X(num_nodes, dim_f), ops
        elif self.type_pool in ['max', 'mean', 'sum']:
            ops = dims_x_l[-1].num_nodes * dims_x_l[-1].num_feats
            mult = 1 if self.type_res == 'none' else len(dims_x_l)
            ops *= mult  # we first pool graph
            dims_root_l = [Dims_X(num_nodes, d.num_feats) for d in dims_x_l]
            _dim_f, ops_res = self.f_res_complexity(dims_root_l)
            ops += 2 * ops_res  # "2" since one for neighs, and the other for root
        elif self.type_pool == 'sort':
            if self.type_res == 'none':
                ops = 0
            else:
                _dim, ops = self.f_res_complexity(dims_x_l)
            # global_sort_pool: sort is only alone last channel, therefore neglect its complexity
            for n in self.nn_pool:
                if type(n) == nn.Linear:
                    ops += np.pool(list(n.weight.shape)) * num_nodes
            ops += np.prod(list(self.nn_pool.weight.shape)) * self.k * num_nodes
        for n in self.nn:
            if type(n) == nn.Linear:
                ops += np.prod(list(n.weight.shape)) * num_nodes
                dim_f = n.weight.shape[0]  # dim 0 is output dim
        return Dims_X(num_nodes, dim_f), ops

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

    def complexity(self, dims_x_l):
        ops = 0
        for dx in dims_x_l:
            assert dx.num_feats == self.f_lin.weight.shape[1]
            ops += dx.num_nodes * dx.num_feats * self.f_lin.weight.shape[0]    # X W
            ops += dx.num_nodes * dx.num_feats      # X q
            ops += dx.num_nodes * dx.num_feats      # sum X[i]
        return Dims_X(dx.num_nodes, self.f_lin.weight.shape[0]), ops

class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.0, act="relu", **kwargs):
        super().__init__()
        if "aggr" in kwargs:
            assert kwargs["aggr"] == "gcn"
        self.dim_in, self.dim_out = dim_in, dim_out
        self.dropout = dropout
        self.act = get_torch_act(act, locals())
        self.f_lin = nn.Linear(dim_in, dim_out, bias=True)
        self.f_dropout = nn.Dropout(p=self.dropout)
        self.offset = nn.Parameter(torch.zeros(dim_out))
        self.scale = nn.Parameter(torch.ones(dim_out))
    
    def f_norm(self, feat_in):
        mean = feat_in.mean(dim=1).view(feat_in.shape[0], 1)
        var = feat_in.var(dim=1, unbiased=False).view(feat_in.shape[0], 1) + 1e-10
        feat_norm = (feat_in - mean) * self.scale * torch.rsqrt(var) + self.offset
        return feat_norm

    def forward(self, inputs):
        feat_in, adj, is_normed, dropedge = inputs
        feat_in = self.f_dropout(feat_in)
        if not is_normed and adj is not None:
            assert type(adj) == sp.csr_matrix
            adj_norm = adj_norm_sym(adj, dropedge=dropedge)     # self-edges are already added by C++ sampler
            adj_norm = coo_scipy2torch(adj_norm.tocoo()).to(feat_in.device)
        else:
            assert adj is None or type(adj) == torch.Tensor
            adj_norm = adj
        feat_aggr = torch.sparse.mm(adj_norm, feat_in)
        feat_trans = self.f_lin(feat_aggr)
        feat_out = self.f_norm(self.act(feat_trans))
        return feat_out, adj_norm, True, 0.

class GraphSAGE(nn.Module):

    def __init__(
        self, dim_in, dim_out, dropout=0.0, act="relu", **kwargs
    ):
        super().__init__()
        self.act = get_torch_act(act, locals())
        self.dropout = dropout
        self.f_lin = []
        self.offset, self.scale = [], []
        self.f_lin_self = nn.Linear(dim_in, dim_out)
        self.f_lin_neigh = nn.Linear(dim_in, dim_out)
        self.offset = nn.Parameter(torch.zeros(dim_out * 2))
        self.scale  = nn.Parameter(torch.ones(dim_out * 2))
        self.f_dropout = nn.Dropout(p=self.dropout)
        self.dim_out = dim_out

    def _spmm(self, adj_norm, _feat):
        """ sparse feature matrix multiply dense feature matrix """
        return torch.sparse.mm(adj_norm, _feat)

    def _f_norm(self, _feat, _id):
        mean = _feat.mean(dim=1).view(_feat.shape[0], 1)
        var = _feat.var(dim=1, unbiased=False).view(_feat.shape[0], 1) + 1e-9
        _scale = self.scale[_id * self.dim_out : (_id + 1) * self.dim_out]
        _offset = self.offset[_id * self.dim_out : (_id + 1) * self.dim_out]
        feat_out = (_feat - mean) * _scale * torch.rsqrt(var) + _offset
        return feat_out

    def forward(self, inputs):
        """
        Inputs:
            adj_norm        normalized adj matrix of the subgraph
            feat_in         2D matrix of input node features

        Outputs:
            adj_norm        same as input (to facilitate nn.Sequential)
            feat_out        2D matrix of output node features
        """
        # dropout-act-norm
        feat_in, adj, is_normed, dropedge = inputs
        if not is_normed and adj is not None:
            assert type(adj) == sp.csr_matrix
            adj = coo_scipy2torch(adj.tocoo()).to(feat_in.device)
            adj_norm = adj_norm_rw(adj, dropedge=dropedge)
        else:
            assert adj is None or type(adj) == torch.Tensor or type(adj) == tuple
            adj_norm = adj
        feat_in = self.f_dropout(feat_in)
        feat_self = feat_in
        feat_neigh = self._spmm(adj_norm, feat_in)
        feat_self_trans = self._f_norm(self.act(self.f_lin_self(feat_self)), 0)
        feat_neigh_trans = self._f_norm(self.act(self.f_lin_neigh(feat_neigh)), 1)
        feat_out = feat_self_trans + feat_neigh_trans
        return feat_out, adj_norm, True, 0.

    def complexity(self, dims_x, dims_adj):
        assert dims_x.num_nodes == dims_adj.num_nodes
        ops = dims_x.num_nodes * np.product(self.f_lin_self.weight.shape) \
            + dims_adj.num_edges * dims_x.num_feats \
            + dims_x.num_nodes * np.product(self.f_lin_neigh.weight.shape)
        return (Dims_X(dims_x.num_nodes, self.f_lin_self.weight.shape[0]),
                Dims_adj(dims_adj.num_nodes, dims_adj.num_edges)), ops


class GIN(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.0, act="relu", eps=0, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.act = get_torch_act(act, locals())
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_out, bias=True), 
            nn.ReLU(),
            nn.Linear(dim_out, dim_out, bias=True))
        self.f_dropout = nn.Dropout(p=self.dropout)
        self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        self.offset = nn.Parameter(torch.zeros(dim_out))
        self.scale = nn.Parameter(torch.ones(dim_out))

    def f_norm(self, feat_in):
        mean = feat_in.mean(dim=1).view(feat_in.shape[0], 1)
        var = feat_in.var(dim=1, unbiased=False).view(feat_in.shape[0], 1) + 1e-10
        feat_norm = (feat_in - mean) * self.scale * torch.rsqrt(var) + self.offset
        return feat_norm
    
    def forward(self, inputs):
        feat_in, adj, is_normed, dropedge = inputs
        assert not is_normed
        feat_in = self.f_dropout(feat_in)
        if type(adj) == sp.csr_matrix:
            adj = coo_scipy2torch(adj.tocoo()).to(feat_in.device)
            deg_orig = get_deg_torch_sparse(adj)
            masked_indices = torch.floor(torch.rand(int(adj._values().size()[0] * dropedge)) * adj._values().size()[0]).long()
            adj._values()[masked_indices] = 0
            deg_dropped = torch.clamp(get_deg_torch_sparse(adj), min=1)
            rescale = torch.repeat_interleave(deg_orig / deg_dropped, deg_orig.long())
            adj._values()[:] = adj._values() * rescale
        feat_aggr = torch.sparse.mm(adj, feat_in)
        feat_aggr += (1 + self.eps) * feat_in
        feat_out = self.mlp(feat_aggr)
        feat_out = self.f_norm(self.act(feat_out))
        return feat_out, adj, False, 0.

    def complexity(self, dims_x, dims_adj):
        assert dims_x.num_nodes == dims_adj.num_nodes
        ops = dims_adj.num_edges * dims_x.num_feats
        # TODO
        ops += dims_x.num_nodes * dims_x.num_feats      # (1 + eps) * X
        for m in self.mlp:
            breakpoint()       # how to access MLP weight dim?
        return 


class GAT(nn.Module):

    def __init__(
        self, dim_in, dim_out, dropout=0.0, act="relu", mulhead=1, **kwargs
    ):
        super().__init__()
        self.mulhead = mulhead
        self.act = nn.ReLU()
        self.att_act = nn.LeakyReLU(negative_slope=0.2)     # See original GAT paper
        self.dropout = dropout
        assert dim_out % self.mulhead == 0, "invalid output dimension: need to be divisible by mulhead"
        self.dim_slice = int(dim_out / self.mulhead)
        self.f_lin = nn.ModuleList(nn.Linear(dim_in, dim_out, bias=True) for i in range(2))        # neigh + self
        self.offset = nn.Parameter(torch.zeros(2, self.mulhead, self.dim_slice))
        self.scale = nn.Parameter(torch.ones(2, self.mulhead, self.dim_slice))
        self.attention = nn.Parameter(torch.ones(2, self.mulhead, self.dim_slice))
        nn.init.xavier_uniform_(self.attention)        
        self.f_dropout = nn.Dropout(p=self.dropout)

    def _spmm(self, adj_norm, _feat):
        return torch.sparse.mm(adj_norm, _feat)

    def _aggregate_attention(self, adj, feat_neigh, feat_self, attention_self, attention_neigh):
        attention_self = self.att_act(attention_self.mm(feat_self.t())).squeeze()       # num_nodes
        attention_neigh = self.att_act(attention_neigh.mm(feat_neigh.t())).squeeze()    # num_nodes
        val_adj = (attention_self[adj._indices()[0]] + attention_neigh[adj._indices()[1]]) # * adj._values()
        # Compute softmax per neighborhood: substract max for stability in softmax computation
        max_per_row = scatter(val_adj, adj._indices()[0], reduce="max")     # here we may select some entry that will eventually be dropedged. But this is ok. 
        deg = scatter(torch.ones(val_adj.size()).to(feat_neigh.device), adj._indices()[0], reduce="sum")
        val_adj_norm = val_adj - torch.repeat_interleave(max_per_row, deg.long())
        val_adj_exp = torch.exp(val_adj_norm) * adj._values()
        # put coefficient alpha into the adj values
        att_adj = torch.sparse.FloatTensor(adj._indices(), val_adj_exp, torch.Size(adj.shape))
        denom = torch.clamp(scatter(val_adj_exp, adj._indices()[0], reduce="sum"), min=1e-10)
        # aggregate
        ret = self._spmm(att_adj, feat_neigh)
        ret *= 1 / denom.view(-1, 1)
        return ret

    def _adj_norm(self, adj, is_normed, device, dropedge=0):
        """
        Will perform edge dropout only when is_normed == False
        """
        if type(adj) == sp.csr_matrix:
            assert not is_normed
            adj_norm = coo_scipy2torch(adj.tocoo()).to(device)
            # here we don't normalize adj (data = 1,1,1...). In DGL, it is sym normed
            if dropedge > 0:
                masked_indices = torch.floor(torch.rand(int(adj_norm._values().size()[0] * dropedge)) * adj_norm._values().size()[0]).long()
                adj_norm._values()[masked_indices] = 0
        else:
            assert type(adj) == torch.Tensor and is_normed
            adj_norm = adj
        return adj_norm

    def _batch_norm(self, feat_neigh, feat_self):
        f_trans = lambda feat_center, idx: feat_center * self.scale[idx].unsqueeze(0) + self.offset[idx].unsqueeze(0)
        for j in range(self.mulhead):
            mean_self = feat_self[j].mean(dim=1).unsqueeze(1)
            mean_neigh = feat_neigh[j].mean(dim=1).unsqueeze(1)
            var_self = torch.rsqrt(feat_self[j].var(dim=1, unbiased=False).unsqueeze(1) + 1e-9)
            var_neigh = torch.rsqrt(feat_neigh[j].var(dim=1, unbiased=False).unsqueeze(1) + 1e-9)
            feat_self[j]  = f_trans((feat_self[j] - mean_self) * var_self, (0, j))
            feat_neigh[j] = f_trans((feat_neigh[j] - mean_neigh) * var_neigh, (1, j))
        return feat_self, feat_neigh

    def forward(self, inputs):
        feat_in, adj, is_normed, dropedge = inputs
        adj_norm = self._adj_norm(adj, is_normed, feat_in.device, dropedge=dropedge)
        feat_in = self.f_dropout(feat_in)
        # generate A^i X
        N = feat_in.shape[0]
        feat_partial_self  = self.act(self.f_lin[0](feat_in)).view(N, self.mulhead, -1)
        feat_partial_neigh = self.act(self.f_lin[1](feat_in)).view(N, self.mulhead, -1)
        feat_partial_self  = [feat_partial_self[:, t] for t in range(self.mulhead)]
        feat_partial_neigh = [feat_partial_neigh[:, t] for t in range(self.mulhead)]
        for k in range(self.mulhead):
            feat_partial_neigh[k] = self._aggregate_attention(
                adj_norm,
                feat_partial_neigh[k],
                feat_partial_self[k],
                self.attention[0, k].unsqueeze(0),
                self.attention[1, k].unsqueeze(0)
            )
        feat_partial_neigh, feat_partial_self = self._batch_norm(feat_partial_neigh, feat_partial_self)
        feat_partial_self = torch.cat(feat_partial_self, dim=1)
        feat_partial_neigh = torch.cat(feat_partial_neigh, dim=1)
        feat_out = (feat_partial_self + feat_partial_neigh)/2
        return feat_out, adj_norm, True, 0.

    def complexity(self, dims_X, dims_adj):
        assert dims_X.num_nodes == dims_adj.num_nodes
        ops = 0
        # ops for X W
        ops += dims_X.num_nodes * np.product(self.f_lin[0].weight.shape)
        ops += dims_X.num_nodes * np.product(self.f_lin[1].weight.shape)
        # ops for atten vector times (X W)
        ops += dims_X.num_nodes * self.f_lin[0].weight.shape[0]
        ops += dims_X.num_nodes * self.f_lin[1].weight.shape[0]
        for _h in range(self.mulhead):
            ops += dims_adj.num_edges * 2
            # ops for softmax (assume calculating softmax is of cost 20 -- exp and division are much more expensive)
            ops += dims_adj.num_edges * 20
        # ops for weighted aggregation
        ops += dims_adj.num_edges * self.f_lin[1].weight.shape[0]
        return (Dims_X(dims_X.num_nodes, self.f_lin[1].weight.shape[0]),
                Dims_adj(dims_adj.num_nodes, dims_adj.num_edges)), ops

class TemporalAttentionLayer(nn.Module):
    def __init__(self, 
                input_dim, 
                n_heads, 
                num_time_steps, 
                attn_drop, 
                residual):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout 
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        # 1: Add position embeddings to input
        position_inputs = torch.arange(0,self.num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(inputs.device)
        temporal_inputs = inputs + self.position_embeddings[position_inputs] # [N, T, F]

        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2],[0])) # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2],[0])) # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2],[0])) # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1]/self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        
        outputs = torch.matmul(q_, k_.permute(0,2,1)) # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)
        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1) # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2**32+1)
        outputs = torch.where(masks==0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs # [h*N, T, T]
                
        # 5: Dropout on attention weights.
        #if self.training:

        attns = self.attn_dp(outputs)

        outputs = torch.matmul(attns, v_)  # [hN, T, F/h]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0]/self.n_heads), dim=0), dim=2) # [N, T, F]
        
        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs, attns

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs


    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)
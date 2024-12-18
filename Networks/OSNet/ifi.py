import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class ifi_simfpn(nn.Module):
    def __init__(self, ultra_pe=False, pos_dim=40, learn_pe=False, require_grad=False, feat_num=4, feat_dim=256):
        super(ifi_simfpn, self).__init__()
        self.pos_dim = pos_dim
        self.ultra_pe = ultra_pe
        self.learn_pe = learn_pe
        self.feat_num = feat_num
        self.feat_dim = feat_dim

        # Position Encoding Setup
        if learn_pe:
            for level in range(self.feat_num):
                self._update_property('pos'+str(level+1), PositionEmbeddingLearned(self.pos_dim//2))
        elif ultra_pe:
            for level in range(self.feat_num):
                self._update_property('pos'+str(level+1), SpatialEncoding(2, self.pos_dim, require_grad=require_grad))
            self.pos_dim += 2
        else:
            self.pos_dim = 2  # Only use (x, y) coordinates

    def forward(self, x, size, level=0):
        h, w = size
        if self.ultra_pe:
            rel_coord = eval('self.pos'+str(level))(x)
        elif self.learn_pe:
            rel_coord = eval('self.pos'+str(level))(x, [1,1,h,w])
        return rel_coord

    def _update_property(self, property, value):
        setattr(self, property, value)

class PositionEmbeddingLearned(nn.Module):
    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.row_embed = nn.Embedding(200, num_pos_feats)
        self.col_embed = nn.Embedding(200, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x, shape):
        h, w = shape[2], shape[3]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1).view(x.shape[0],h*w, -1)
        return pos

class SpatialEncoding(nn.Module):
    def __init__(self, in_dim, out_dim, sigma=6, cat_input=True, require_grad=False):
        super().__init__()
        assert out_dim % (2*in_dim) == 0, "dimension must be divisible"

        n = out_dim // 2 // in_dim
        m = 2**np.linspace(0, sigma, n)
        m = np.stack([m] + [np.zeros_like(m)]*(in_dim-1), axis=-1)
        m = np.concatenate([np.roll(m, i, axis=-1) for i in range(in_dim)], axis=0)
        self.emb = torch.FloatTensor(m)
        if require_grad:
            self.emb = nn.Parameter(self.emb, requires_grad=True)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        self.cat_input = cat_input
        self.require_grad = require_grad

    def forward(self, x):
        if not self.require_grad:
            self.emb = self.emb.to(x.device)
        y = torch.matmul(x, self.emb.T)
        if self.cat_input:
            return torch.cat([x, torch.sin(y), torch.cos(y)], dim=-1)
        else:
            return torch.cat([torch.sin(y), torch.cos(y)], dim=-1)

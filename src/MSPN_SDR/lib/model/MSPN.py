import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
from .NAF_utils.arch_util import LayerNorm2d
from natten.natten2d import natten2dqkrpb, natten2dav


class MSPNLayer(nn.Module):
    def __init__(self, embed_dim, window_size=7, bias=True):
        super().__init__()

        self.window_size = window_size
        self.pad = self.window_size // 2

        self.norm1 = LayerNorm2d(embed_dim)
        self.conv = nn.Conv2d(in_channels=embed_dim + 1, out_channels=embed_dim * 2, kernel_size=1, padding=0, bias=bias)

        # define a parameter table of relative position bias
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(1, (2 * self.window_size - 1), (2 * self.window_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, g, cd, var):
        g = self.norm1(g)
        z = torch.cat((g, cd), dim=1)
        qk = self.conv(z)
        q, k = qk.chunk(2, dim=1)
        B, C, H, W = q.shape

        confidence = 1/(1+var)
        #confidence = torch.exp(-var)

        query = q.view(B, 1, -1, H, W).permute(0, 1, 3, 4, 2)
        key = (k * confidence).view(B, 1, -1, H, W).permute(0, 1, 3, 4, 2)
        attn = natten2dqkrpb(query, key, self.rpb, kernel_size=self.window_size, dilation=1)
        attn = self.softmax(attn)

        v_out = natten2dav(attn, cd.unsqueeze(-1), kernel_size=self.window_size, dilation=1)
        cd_out = v_out.squeeze().view(cd.shape)

        var_out = natten2dav(attn, (var + cd.square()).unsqueeze(-1), kernel_size=self.window_size, dilation=1)
        var_out = var_out.squeeze().view(var.shape)

        var_out -= cd_out.square()
        
        # Linear Assumption Kalman Filter
        gain = (var + 1e-12) / (var + var_out + 1e-12)

        cd_out = cd + gain * (cd_out - cd)
        var_out = var - gain * var

        return cd_out, var_out, gain


class MSPN(nn.Module):
    def __init__(self, embed_dim, prop_time):
        super().__init__()

        self.embed_dim = embed_dim
        self.window_size = 13
        self.min_prop_time = prop_time
        self.kappa = 2

        self.mspn = MSPNLayer(embed_dim=self.embed_dim, window_size=self.window_size, bias=True)

    def forward(self, pred_init, var_init, guide):
        B, _, Wh, Ww = pred_init.shape

        cd = pred_init
        var = var_init

        list_feat = [cd.contiguous(), ]
        list_var = [var.contiguous(), ]
        list_gain = []

        for pt in range(self.min_prop_time):
            cd, var, gain = self.mspn(guide, cd, var)

            cd_out = cd.contiguous()
            var_out = var.contiguous()
            gain_out = gain.contiguous()

            list_feat.append(cd_out)
            list_var.append(var_out)
            list_gain.append(gain_out)

        return list_feat, list_var, list_gain


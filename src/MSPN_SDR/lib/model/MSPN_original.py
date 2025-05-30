import torch
import torch.nn as nn
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

    def forward(self, g, cd, sd, mask):
        g = self.norm1(g)
        z = torch.cat((g, cd), dim=1)
        qk = self.conv(z)
        q, k = qk.chunk(2, dim=1)
        v = cd

        B, C, H, W = q.shape

        query = q.view(B, 1, -1, H, W).permute(0, 1, 3, 4, 2)
        key = (k * mask).view(B, 1, -1, H, W).permute(0, 1, 3, 4, 2)
        attn = natten2dqkrpb(query, key, self.rpb, kernel_size=self.window_size, dilation=1)
        attn = self.softmax(attn)

        v_out = natten2dav(attn, v.unsqueeze(-1), kernel_size=self.window_size, dilation=1)
        cd_out = v_out.squeeze().view(cd.shape)
        cd_out = cd_out * mask + cd * (1 - mask)
        cd_out[sd > 0] = sd[sd > 0]

        mask_out = natten2dav(attn, mask.unsqueeze(-1), kernel_size=self.window_size, dilation=1)
        mask_out = mask_out.squeeze().view(mask.shape)
        mask_out[sd > 0] = mask[sd > 0]

        return cd_out, mask_out


class MSPN(nn.Module):
    def __init__(self, embed_dim=64, prop_time=6):
        super().__init__()

        self.embed_dim = embed_dim
        self.window_size = 13
        self.min_prop_time = prop_time
        self.kappa = 2

        self.mspn = MSPNLayer(embed_dim=self.embed_dim, window_size=self.window_size, bias=True)
        self.mspn_2nd = MSPNLayer(embed_dim=self.embed_dim, window_size=self.window_size, bias=True)

    def forward(self, pred_init, list_feat, list_mask, guide, dep, mask_init):
        B, _, Wh, Ww = pred_init.shape

        cd = pred_init
        mask = mask_init

        prop_times_1st = 6

        for pt in range(prop_times_1st):
            cd, mask = self.mspn(guide, cd, dep, mask)

            cd_out = cd.contiguous()
            list_feat.append(cd_out)
            mask_out = mask.contiguous()
            list_mask.append(mask_out)

        mask = mask_init
        cd[dep > 0] = dep[dep > 0]
        for pt in range(6):
            cd, mask = self.mspn_2nd(guide, cd, dep, mask)

            cd_out = cd.contiguous()
            list_feat.append(cd_out)
            mask_out = mask.contiguous()
            list_mask.append(mask_out)

        return cd_out, list_feat, list_mask

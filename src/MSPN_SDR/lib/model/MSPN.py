import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
from .NAF_utils.arch_util import LayerNorm2d
from natten.natten2d import natten2dqkrpb, natten2dav


class MSPNLayer(nn.Module):
    def __init__(self, args, embed_dim, window_size=7, bias=True):
        super().__init__()
        self.args = args

        self.window_size = window_size
        self.pad = self.window_size // 2

        self.norm1 = LayerNorm2d(embed_dim)
        self.conv = nn.Conv2d(in_channels=embed_dim + 1, out_channels=embed_dim * 2, kernel_size=1, padding=0, bias=bias)
        self.conv12 = nn.Conv2d(1, 2, kernel_size=1)
        self.conv21 = nn.Conv2d(2, 1, kernel_size=1)
        # define a parameter table of relative position bias
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(1, (2 * self.window_size - 1), (2 * self.window_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)

        self.softmax = nn.Softmax(dim=-1)

        #input: torch.Size([1, 235, 426, 560])
        self.variance_net = nn.Sequential(
            nn.Conv2d(235, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # nn.DropOut(0.1)
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.ReLU(),
        )
        #output: torch.Size([1, 1, 426, 560])

    def forward(self, g, cd, var):
        g = self.norm1(g)
        z = torch.cat((g, cd), dim=1)
        qk = self.conv(z)
        q, k = qk.chunk(2, dim=1)
        B, C, H, W = q.shape

        #confidence = 1/(1+var)
        confidence = torch.exp(-var)


        query = q.view(B, 1, -1, H, W).permute(0, 1, 3, 4, 2)
        key = (k * confidence).view(B, 1, -1, H, W).permute(0, 1, 3, 4, 2)
        attn = natten2dqkrpb(query, key, self.rpb, kernel_size=self.window_size, dilation=1)
        attn = self.softmax(attn)

        v_out = natten2dav(attn, cd.unsqueeze(-1), kernel_size=self.window_size, dilation=1)
        cd_out = v_out.squeeze().view(cd.shape)

        #var_out = natten2dav(attn, var.unsqueeze(-1), kernel_size=self.window_size, dilation=1)
        #var_out = var_out.squeeze().view(var.shape)

        cd = self.conv21(cd)

        var_input = torch.cat((cd, var, g, attn.squeeze(1).permute(0, -1, 1, 2)), dim=1)
        var_out = self.variance_net(var_input)
        var_out = var_out.squeeze().view(var.shape)
        
        # Linear Assumption Kalman Filter
        gain = var / (var + var_out + 1e-8)
        #print("gain: ", gain.min().item(), gain.mean().item(), gain.max().item())
        cd_out = cd + gain * (cd_out - cd)
        var_out = (1 - gain) * var

        #cd_out[sd > 0] = cd[sd > 0]
        #var_out[sd > 0] = var[sd > 0]
        return cd_out, var_out


class MSPN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.embed_dim = args.embed_dim
        self.window_size = 13
        self.min_prop_time = args.prop_time
        self.kappa = 2

        self.mspn = MSPNLayer(self.args, embed_dim=self.embed_dim, window_size=self.window_size, bias=True)
        if args.mode == 'SDR':
            self.mspn_2nd = MSPNLayer(self.args, embed_dim=self.embed_dim, window_size=self.window_size, bias=True)

    def set_prop_times(self, sd, num_samples):
        B, C, H, W = sd.shape

        avg_dist = (H * W / num_samples) ** 0.5 - 1
        min_iter = torch.floor(avg_dist * self.kappa / (self.window_size // 2)) + 1

        prop_time = []
        for b in range(B):
            prop_time.append(int(max(self.min_prop_time, min_iter[b])))

        return prop_time

    def forward(self, pred_init, list_feat, list_var, guide, dep, mask_init, num_samples=None, mask=None):
        B, _, Wh, Ww = pred_init.shape

        cd = pred_init
        #cd.requires_grad = True
        if mask is None:
            mask = mask_init

        prop_times = self.set_prop_times(dep, num_samples)
        prop_times_1st = max(prop_times)

        for pt in range(prop_times_1st):
            cd, mask = self.mspn(guide, cd, mask)
            cd_out = cd.contiguous()
            list_feat.append(cd_out)
            var_out = mask.contiguous()
            list_var.append(var_out)

        #if self.args.mode == 'SDR':
        #    mask = mask_init
        #    #cd[dep > 0] = dep[dep > 0]
        #    for pt in range(6):
        #        cd, mask = self.mspn_2nd(guide, cd, dep, mask)

        #        cd_out = cd.contiguous()
        #        list_feat.append(cd_out)
        #        mask_out = mask.contiguous()
        #        list_mask.append(mask_out)

        return cd_out, list_feat, list_var


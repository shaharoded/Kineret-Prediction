"""STraTS-style transformer baseline for sparse irregular clinical events.

STraTS represents each observation as a triplet containing measurement value,
timestamp, and variable identity. Continuous value/time embeddings and variable
embeddings are fused, then a transformer encoder models interactions across the
set of irregular observations rather than forcing the trajectory onto a dense
regular grid.

In this study it is the **ss-STraTS** configuration -- supervised only, no
self-supervised forecasting stage. The encoded observation window is combined
with the static context vector (which carries the aggregated QA compliance block
in the QA arm) and trained with a multi-label risk head plus per-outcome onset
and length-of-stay regressions.

Reference:
- Tipirneni and Reddy, "Self-supervised transformer for sparse and irregularly
  sampled multivariate clinical time-series" (TKDD, 2022):
  https://github.com/sindhura97/STraTS
"""

from argparse import Namespace
from kineret.strats.model_utils import TimeSeriesModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np



class CVE(nn.Module):
    def __init__(self, args):
        super().__init__()
        int_dim = int(np.sqrt(args.hid_dim))
        self.W1 = nn.Parameter(torch.empty(1,int_dim), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.W2 = nn.Parameter(torch.empty(int_dim, args.hid_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.activation = torch.tanh

    def forward(self, x):
        # x: bsz, max_len
        x = torch.unsqueeze(x, -1)
        x = torch.matmul(x,self.W1) + self.b1[None,None,:] # bsz,max_len,int_dim
        x = self.activation(x)
        x = torch.matmul(x,self.W2) #bsz,max_len,hid_dim
        return x


class FusionAtt(nn.Module):
    def __init__(self, args):
        super().__init__()
        int_dim = args.hid_dim
        self.W = nn.Parameter(torch.empty(args.hid_dim,int_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.u = nn.Parameter(torch.empty(int_dim,1), requires_grad=True)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.u)
        self.activation = torch.tanh

    def forward(self, x, mask):
        # x: bsz, max_len, hid_dim
        att = torch.matmul(x,self.W) + self.b[None,None,:] # bsz,max_len,int_dim
        att = self.activation(att)
        att = torch.matmul(att,self.u)[:,:,0] # bsz,max_len
        att = att + (1-mask)*torch.finfo(att.dtype).min
        att = torch.softmax(att,dim=-1) # bsz,max_len
        return att


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.N = args.num_layers
        self.d = args.hid_dim
        self.dff = self.d*2
        self.attention_dropout = args.attention_dropout
        self.dropout = args.dropout
        self.h = args.num_heads
        self.dk = self.d//self.h
        self.all_head_size = self.dk*self.h

        self.Wq = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wk = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wv = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wo = nn.Parameter(self.init_proj((self.N, self.all_head_size, self.d)), requires_grad=True)
        self.W1 = nn.Parameter(self.init_proj((self.N, self.d, self.dff)), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros((self.N, 1, 1, self.dff)), requires_grad=True)
        self.W2 = nn.Parameter(self.init_proj((self.N, self.dff, self.d)), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros((self.N, 1, 1, self.d)), requires_grad=True)
        # self.layer_norm1 = nn.ModuleList([nn.LayerNorm(self.d) for i in range(self.N)])
        # self.layer_norm2 = nn.ModuleList([nn.LayerNorm(self.d) for i in range(self.N)])

    def init_proj(self, shape, gain=1):
        x = torch.rand(shape)
        fan_in_out = shape[-1]+shape[-2]
        scale = gain*np.sqrt(6/fan_in_out)
        x = x*2*scale - scale
        return x

    def forward(self, x, mask):
        # x: bsz, max_len, d
        # mask: bsz, max_len
        bsz, max_len, _ = x.size()
        mask = mask[:,:,None]*mask[:,None,:]
        mask = (1-mask)[:,None,:,:]*torch.finfo(x.dtype).min
        layer_mask = mask
        for i in range(self.N):
            # MHA
            q = torch.einsum('bld,hde->bhle', x, self.Wq[i]) # bsz,h,max_len,dk
            k = torch.einsum('bld,hde->bhle', x, self.Wk[i]) # bsz,h,max_len,dk
            v = torch.einsum('bld,hde->bhle', x, self.Wv[i]) # bsz,h,max_len,dk
            # x_for_qkv = x[:,None,:,:,None]
            # q = (x_for_qkv*self.Wq[i][None,:,None,:,:]).sum(dim=-2) # bsz,h,max_len,dk
            # k = (x_for_qkv*self.Wk[i][None,:,None,:,:]).sum(dim=-2) # bsz,h,max_len,dk
            # v = (x_for_qkv*self.Wv[i][None,:,None,:,:]).sum(dim=-2) # bsz,h,max_len,dk
            A = torch.einsum('bhle,bhke->bhlk', q, k) # bsz,h,max_len,max_len
            # A = (q[:,:,:,None,:]*k[:,:,None,:,:]).sum(dim=-1) # bsz,h,max_len,max_len
            if self.training:
                dropout_mask = (torch.rand_like(A)<self.attention_dropout
                                ).float()*torch.finfo(x.dtype).min
                layer_mask = mask+dropout_mask
            A = A+layer_mask
            A = torch.softmax(A, dim=-1)
            v = torch.einsum('bhkl,bhle->bkhe',A,v) # bsz,max_len,h,dk
            # v = (A[:,:,:,:,None]*v[:,:,None,:,:]).sum(dim=-2).transpose(1,2) # bsz,max_len,h,dk
            all_head_op = v.reshape((bsz, max_len, -1))
            all_head_op = torch.matmul(all_head_op, self.Wo[i])
            all_head_op = F.dropout(all_head_op, self.dropout, self.training)
            # Add+layernorm
            # x = self.layer_norm1[i](all_head_op+x) # bsz,max_len,d
            x = (all_head_op+x)/2
            # FFN
            ffn_op = torch.matmul(x,self.W1[i])+self.b1[i]
            ffn_op = F.gelu(ffn_op)
            ffn_op = torch.matmul(ffn_op,self.W2[i])+self.b2[i]
            ffn_op = F.dropout(ffn_op, self.dropout, self.training)
            # Add+layernorm
            # x = self.layer_norm2[i](ffn_op+x)
            x = (ffn_op+x)/2
        return x




class Strats(TimeSeriesModel):
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.cve_time = CVE(args)
        self.cve_value = CVE(args)
        self.variable_emb = nn.Embedding(args.V, args.hid_dim)
        self.transformer = Transformer(args)
        self.fusion_att = FusionAtt(args)
        self.dropout = args.dropout
        self.V = args.V


    def forward(self, values, times, varis, obs_mask, demo,
                labels=None, los_target_norm=None, los_mask=None,
                time_target_norm=None, time_mask=None):
        bsz, max_obs = values.size()
        device = values.device
        if self.training:
            with torch.no_grad():
                var_mask = (torch.rand((bsz,self.V), device=device)<=self.dropout).int()
                for v in range(self.V):
                    mask_pos = (varis==v).int()*var_mask[:,v:v+1]
                    obs_mask = obs_mask*(1-mask_pos)

        # Static context, projected. In the QA arm this vector already carries
        # the aggregated QA_<pattern> compliance block.
        demo_emb = self.demo_emb(demo)
        # initial triplet embedding
        time_emb = self.cve_time(times)
        value_emb = self.cve_value(values)
        # value_emb = 0
        # for i in range(self.args.V):
        #     value_emb = value_emb + self.cve_value[i](values) * (varis==i)
        vari_emb = self.variable_emb(varis)
        triplet_emb = time_emb+value_emb+vari_emb
        triplet_emb = F.dropout(triplet_emb, self.dropout, self.training)
        # contextual triplet emb
        contextual_emb = self.transformer(triplet_emb, obs_mask)
        # fusion attention
        attention_weights = self.fusion_att(contextual_emb, obs_mask)[:,:,None]
        ts_emb = (contextual_emb * attention_weights).sum(dim=1)
        # concat demo and ts_emb
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        logits = self.binary_head(ts_demo_emb)
        return self.binary_cls_final(logits, labels, los_target_norm, los_mask,
                                     time_target_norm, time_mask)

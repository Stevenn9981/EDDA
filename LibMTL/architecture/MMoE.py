import pdb
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.architecture.abstract_arch import AbsArchitecture


class MMoE(AbsArchitecture):
    r"""Multi-gate Mixture-of-Experts (MMoE).
    
    This method is proposed in `Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts (KDD 2018) <https://dl.acm.org/doi/10.1145/3219819.3220007>`_ \
    and implemented by us.

    Args:
        img_size (list): The size of input data. For example, [3, 244, 244] for input images with size 3x224x224.
        num_experts (int): The number of experts shared for all tasks. Each expert is the encoder network.

    """

    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(MMoE, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)

        self.input_size = self.kwargs['input_size']
        # self.num_experts = self.kwargs['num_experts'][0]
        self.num_experts = len(self.task_name)
        self.experts_shared = nn.ModuleList([encoder_class() for _ in range(self.num_experts)])
        self.att = self.kwargs['mmoe_att']
        if self.att:
            self.attention = nn.MultiheadAttention(self.kwargs['hidden_dim'], 8)
            query_vectors = torch.randn((self.num_experts, self.kwargs['hidden_dim']), device=device)
            self.queries = {task: query_vectors[i] for i, task in enumerate(self.task_name)}
        else:
            self.gate_specific = nn.ModuleDict(
                {task: nn.Sequential(nn.Linear(self.kwargs['hidden_dim'] * 2, self.num_experts),
                                     nn.Softmax(dim=-1)) for task in self.task_name})
        self.node_nums = self.kwargs['node_nums']
        self.user_emb = nn.ParameterDict({
            task: nn.Parameter(torch.randn(self.node_nums['user'][task], self.kwargs['hidden_dim'])) for task in
            self.task_name
        })
        self.item_emb = nn.ParameterDict({
            task: nn.Parameter(torch.randn(self.node_nums['item'][task], self.kwargs['hidden_dim'])) for task in
            self.task_name
        })
        for task in self.task_name:
            nn.init.normal_(self.user_emb[task], std=0.1)
            nn.init.normal_(self.item_emb[task], std=0.1)
        self.f = nn.Sigmoid()
        # self.queries = nn.ParameterDict(
        #     {task: nn.Parameter(query_vectors[i]) for i, task in enumerate(self.task_name)})
        # self.keys = nn.ParameterDict(
        #     {task: nn.Parameter(domains[i]) for i, task in enumerate(self.task_name)})

    def forward(self, inputs, task_name=None):
        input = torch.cat([self.user_emb[task_name][inputs[0]], self.item_emb[task_name][inputs[1]]], dim=1)
        experts_shared_rep = torch.stack([e(input) for e in self.experts_shared])
        out = {}
        for task in self.task_name:
            if task_name is not None and task != task_name:
                continue
            if self.att:
                query = self.queries[task].expand(1, experts_shared_rep.shape[1], -1)
                att_output, _ = self.attention(query, experts_shared_rep, experts_shared_rep, need_weights=False)
                gate_rep = att_output.squeeze(0)
            else:
                selector = self.gate_specific[task](torch.flatten(input, start_dim=1))
                gate_rep = torch.einsum('ij..., ji -> j...', experts_shared_rep, selector)
            self._prepare_rep(gate_rep, task, same_rep=False)
            if gate_rep.device == 'cpu':
                trans = self.decoders[task].to('cpu')
                out[task] = trans(gate_rep)
            else:
                out[task] = self.decoders[task](gate_rep)
            # out[task] = self.f(out[task])
        return out

    def get_share_params(self):
        return self.experts_shared.parameters()

    def zero_grad_share_params(self):
        self.experts_shared.zero_grad()

    # def get_middle_embedding(self, inputs, task_name):
    #     experts_shared_rep = torch.stack([e(inputs) for e in self.experts_shared])
    #     # selector = self.gate_specific[task_name](torch.flatten(inputs, start_dim=1))
    #     # gate_rep = torch.einsum('ij..., ji -> j...', experts_shared_rep, selector).detach()
    #
    #     query = self.queries[task_name].expand(1, experts_shared_rep.shape[1], -1)
    #     att_output, _ = self.attention(query, experts_shared_rep, experts_shared_rep, need_weights=False)
    #     gate_rep = att_output.squeeze(0)
    #
    #     # gate_rep = torch.pca_lowrank(gate_rep, 8)[0]
    #     gate_rep = self.decoders[task_name][0](gate_rep)
    #     return gate_rep

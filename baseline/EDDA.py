"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import pdb
import time

import dgl
import torch
from torch import nn, optim
import torch.nn.functional as F

from baseline.GRecConv import GRecConv
from task_dataloader import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class EDDA(nn.Module):
    def __init__(self, tasks, hidden_dim=64, sim_dict=None, graph_dict=None, edge_dropout=0.3):
        super(EDDA, self).__init__()
        self.domain_id_map = dict()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.latent_dim = hidden_dim
        self.n_layers = 2
        self.domains = tasks
        self.sim_dict = sim_dict
        self.f = nn.Sigmoid()
        self.aggr_user = torch.nn.Embedding(
            num_embeddings=len(TOTAL_USER_ID_DICT), embedding_dim=hidden_dim // 2)
        self.aggr_item = torch.nn.Embedding(
            num_embeddings=len(TOTAL_ITEM_ID_DICT), embedding_dim=hidden_dim // 2)
        nn.init.normal_(self.aggr_user.weight, std=0.1)
        nn.init.normal_(self.aggr_item.weight, std=0.1)
        self.conv = GRecConv(layer=self.n_layers, alpha=0.1, edge_drop=edge_dropout)
        user_embList = []
        item_embList = []
        map_func = dict()
        for i, task in enumerate(tasks):
            self.domain_id_map[task] = i
            map_func[task] = nn.Linear(hidden_dim // 2, hidden_dim // 2, bias=False)
            user_embList.append(torch.nn.Embedding(
                num_embeddings=len(USER_DICT[task]), embedding_dim=hidden_dim // 2))
            item_embList.append(torch.nn.Embedding(
                num_embeddings=len(ITEM_DICT[task]), embedding_dim=hidden_dim // 2))
            nn.init.normal_(user_embList[i].weight, std=0.1)
            nn.init.normal_(item_embList[i].weight, std=0.1)
        self.map_func = nn.ModuleDict(map_func)
        self.embedding_user = nn.ModuleList(user_embList)
        self.embedding_item = nn.ModuleList(item_embList)
        self.edge_dropout = edge_dropout
        self.graph_dict = graph_dict
        for task in tasks:
            self.graph_dict['sep' + task] = dgl.to_bidirected(
                dgl.in_subgraph(self.graph_dict['aggr'].to('cpu'),
                                ITEM_DOMAIN_DICT[task] + len(TOTAL_USER_ID_DICT))).to(self.device)
        self.cos_fn = nn.CosineSimilarity()
        self.dis_fn = nn.PairwiseDistance(p=2)
        self.ablation = False

    def computer(self, graph, domain, mode='normal'):

        if mode == 'dis_inter':
            users_emb = self.aggr_user.weight
            items_emb = self.aggr_item.weight
            layer_emb = torch.cat([users_emb, items_emb])
        elif mode == 'dis_intra':
            if self.ablation:
                users_emb = self.aggr_user.weight[USER_DOMAIN_DICT[domain]]
                items_emb = self.aggr_item.weight[ITEM_DOMAIN_DICT[domain]]
            else:
                users_emb = self.embedding_user[self.domain_id_map[domain]].weight
                items_emb = self.embedding_item[self.domain_id_map[domain]].weight
            layer_emb = torch.cat([users_emb, items_emb])
        else:
            user_intra, item_intra = self.computer(self.graph_dict[domain], domain, mode='dis_intra')
            user_inter, item_inter = self.computer(self.graph_dict['aggr'], domain, mode='dis_inter')
            # user_inter, item_inter = user_inter[USER_DOMAIN_DICT[domain]], item_inter[ITEM_DOMAIN_DICT[domain]]

            # import pdb;pdb.set_trace()
            users_emb = torch.cat([user_inter, user_intra], dim=1)
            items_emb = torch.cat([item_inter, item_intra], dim=1)

            return users_emb, items_emb

        if mode == 'dis_inter':
            embs = []
            for dom in self.domains:
                t_emb = layer_emb
                # for _ in range(self.n_layers):
                graph = self.graph_dict['sep' + dom]
                t_emb = self.conv(graph, t_emb)
                uss, its = torch.split(t_emb, [users_emb.shape[0], items_emb.shape[0]])
                uss, its = uss[USER_DOMAIN_DICT[domain]], its[ITEM_DOMAIN_DICT[domain]]
                t_emb = torch.cat([uss, its], dim=0)
                embs.append(t_emb)
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
        else:
            layer_emb = self.conv(graph, layer_emb)
            light_out = layer_emb

        users, items = torch.split(light_out, [len(USER_DOMAIN_DICT[domain]), len(ITEM_DOMAIN_DICT[domain])])
        return users, items

    def getUsersRating(self, users, items, graph, domain, mode='normal'):
        all_users, all_items = self.computer(graph, domain, mode)
        users_emb = all_users[users.long()]
        items_emb = all_items[items.long()]
        rating = torch.mul(users_emb, items_emb).sum(1)
        return rating

    def getEmbedding(self, graph, users, domain, pos_items, neg_items, mode='normal'):
        all_users, all_items = self.computer(graph, domain, mode)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        if self.ablation:
            users_emb1 = self.aggr_user.weight[USER_DOMAIN_DICT[domain]]
            items_emb1 = self.aggr_item.weight[ITEM_DOMAIN_DICT[domain]]
        else:
            users_emb1 = self.embedding_user[self.domain_id_map[domain]].weight
            items_emb1 = self.embedding_item[self.domain_id_map[domain]].weight
        users_emb2 = self.aggr_user.weight[USER_DOMAIN_DICT[domain]]
        items_emb2 = self.aggr_item.weight[ITEM_DOMAIN_DICT[domain]]
        user_emb = torch.cat([users_emb1, users_emb2], dim=1)
        item_emb = torch.cat([items_emb1, items_emb2], dim=1)
        users_emb_ego = user_emb[users]
        pos_emb_ego = item_emb[pos_items]
        neg_emb_ego = item_emb[neg_items]

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, graph, users, pos, neg, domain, mode='normal'):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(graph, users.long(), domain, pos.long(), neg.long(), mode)
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def cal_discrepancy_loss(self, domain):
        users_emb1 = self.embedding_user[self.domain_id_map[domain]].weight
        items_emb1 = self.embedding_item[self.domain_id_map[domain]].weight
        users_emb2 = self.aggr_user.weight[USER_DOMAIN_DICT[domain]]
        items_emb2 = self.aggr_item.weight[ITEM_DOMAIN_DICT[domain]]
        loss = torch.mean(self.cos_fn(users_emb1, users_emb2)) + torch.mean(self.cos_fn(items_emb1, items_emb2))
        return loss

    def cal_align_loss(self, domain1, domain2, pairs):
        if self.ablation:
            users_emb = self.aggr_user.weight[USER_DOMAIN_DICT[domain1]]
            items_emb = self.aggr_item.weight[ITEM_DOMAIN_DICT[domain1]]
        else:
            users_emb = self.embedding_user[self.domain_id_map[domain1]].weight
            items_emb = self.embedding_item[self.domain_id_map[domain1]].weight
        d1_emb = torch.cat([users_emb, items_emb])

        if self.ablation:
            users_emb = self.aggr_user.weight[USER_DOMAIN_DICT[domain2]]
            items_emb = self.aggr_item.weight[ITEM_DOMAIN_DICT[domain2]]
        else:
            users_emb = self.embedding_user[self.domain_id_map[domain2]].weight
            items_emb = self.embedding_item[self.domain_id_map[domain2]].weight
        d2_emb = torch.cat([users_emb, items_emb])

        d1_emb, d2_emb = self.map_func[domain1](d1_emb), self.map_func[domain2](d2_emb)
        align_loss = torch.mean(self.dis_fn(d1_emb[pairs[0]], d2_emb[pairs[1]]))

        return align_loss

    def forward(self, graph, users, items, domain):
        # compute embedding
        all_users, all_items = self.computer(graph, domain)
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math
import networkx as nx
import pandas as pd
from util import *

import numpy as np

class AGCNBlock(nn.Module):

    def __init__(self,config,input_dim,hidden_dim,gcn_layer=2,dropout=0.0,relu=0):
        super(AGCNBlock,self).__init__()
        self.percent = config.percent

        if config.pool=='mean':
            self.pool=self.mean_pool
        elif config.pool=='max':
            self.pool=self.max_pool
        elif config.pool=='sum':
            self.pool=self.sum_pool

        self.eps = config.eps
        self.num_layers = config.num_layers
        self.lamda1 = config.lamda1
        self.lamda2 = config.lamda2
        self.sele_method = config.sele_method
        self.proj_mode = config.proj_mode
        self.distance_mode = config.distance_mode
        self.mode = config.mode


        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.model=config.model
        self.aggre_mode = config.aggre_mode
        self.struct_mode = config.struct_mode
        self.gcns=nn.ModuleList()

        # hidden_dim = input_dim
        self.pass_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.W = nn.Parameter(torch.zeros(1, hidden_dim, hidden_dim))
        # torch.nn.init.uniform_(self.W)
        torch.nn.init.xavier_uniform_(self.W)

        self.gcns = nn.ModuleList()
        self.gcns.append(GCNBlock(input_dim, hidden_dim, config,config.bn, config.gcn_res, config.gcn_norm, dropout, relu))

        for i in range(gcn_layer - 1):
            if i == gcn_layer - 2 and (not config.lastrelu):
                self.gcns.append(
                    GCNBlock(hidden_dim, hidden_dim, config, config.bn, config.gcn_res, config.gcn_norm, dropout, 0))
            else:
                self.gcns.append(
                    GCNBlock(hidden_dim, hidden_dim, config, config.bn, config.gcn_res, config.gcn_norm, dropout, relu))

    def mean_pool(self, x, mask):
        return x.sum(dim=1) / (self.eps + mask.sum(dim=1, keepdim=True))

    def sum_pool(self, x, mask):
        return x.sum(dim=1)

    def max_pool(self, x,mask):
        #output: [batch,x.shape[2]]
        m=(mask-1)*1e10
        r,_=(x+m.unsqueeze(2)).max(dim=1)
        return r

    def find_delta(self, mask, scores, hops):

        delta = scores.new_zeros(scores.shape).type(torch.FloatTensor)
        for i in range(int(mask.shape[0])):
            cur_node_num = int(mask[i].sum())
            score = scores[i, 0:cur_node_num]
            max_idx = -1
            for j in range(cur_node_num):

                score_tmp = float(score[j])
                large_idx = torch.arange(0, score.numel())[score.gt(score_tmp)]

                if len(large_idx) == 0:
                    score[j] = -1
                    large_idx = torch.arange(0, score.numel())[score.eq(score_tmp)]
                    score[j] = score_tmp
                if len(large_idx) == 0:
                    max_idx = j
                else:
                    sele_dis = hops[i][j][large_idx]

                    min_dis = float(torch.min(sele_dis))
                    delta[i][j] = float(torch.min(sele_dis))
            if max_idx != -1:
                delta[i][max_idx] = 1.



        return delta

    def re_arange(self, nodes, min_hops):
        arange_hops = torch.zeros(min_hops.shape)
        for k in range(len(nodes)):
            arange_hops[k] = min_hops[nodes[k]]
        return arange_hops

    def find_top_k(self, score, hops, k_max, k_list):
        batch_size = score.shape[0]
        top_index = score.new_zeros(batch_size, k_max).type(torch.LongTensor)

        for i in range(batch_size):

            sele_idx_out = []
            mask = hops.new_zeros(hops.shape[1])

            max_deg_idx = int(torch.max(score[i], 0)[1])
            max_deg_val = torch.max(score[i], 0)[0]

            sele_idx_out.append(max_deg_idx)
            mask[max_deg_idx] = -1

            for j in range(1, k_list[i]):
                tmp_sele = torch.LongTensor(sele_idx_out)

                tmp_hops = hops[i][tmp_sele]+ mask.repeat(len(sele_idx_out), 1)
                min_hops, min_idx = torch.min(tmp_hops, 0)

                gamma = torch.mul(score[i], min_hops)
                top_val, top_idx = torch.max(gamma, 0)

                mask[top_idx] = -1
                sele_idx_out.append(int(top_idx))

            top_index[i, 0:k_list[i]] = torch.LongTensor(sele_idx_out)

        return top_index

    def find_new_hop(self, new_adj, new_mask, graph, top_index, label, layer_num):

        new_g = nx.Graph()
        new_g_w = nx.Graph()
        for j in range(int(new_mask)):
            cur_idx = list(graph.g.nodes())[top_index[j]]
            new_g.add_node(cur_idx)
            new_g_w.add_node(cur_idx)
            for k in range(int(new_mask)):
                if new_adj[j][k] > 0 and j!= k:
                    new_g.add_edge(cur_idx, list(graph.g.nodes())[top_index[k]])
                    new_g_w.add_edge(cur_idx, list(graph.g.nodes())[top_index[k]], weight = float(torch.exp(new_adj[j][k].neg())))

        hg = Hyper_Graph(new_g,label=label)
        hg_w = Hyper_Graph(new_g_w, label=label)


        _, nor_hops = hg.minimal_distance(hg.g, hg.node_idx)
        _, nor_hops_w = hg.minimal_distance(hg_w.g, hg_w.node_idx)

        return nor_hops+nor_hops_w, hg
        # print(1)



    def forward(self, X,adj,mask, nor_hops,layer_num,epoch,pos, is_print=False):

        hidden = X

        k_max = int(math.ceil(self.percent * adj.shape[-1]))
        k_list = [int(math.ceil(self.percent * x)) for x in mask.sum(dim=1).tolist()]
        new_mask = X.new_zeros(X.shape[0], k_max)
        new_mask_assi = mask.new_ones(X.shape[0], k_max)*(-100000000)


        diag = torch.diagonal(adj, 0, 1, 2)
        if self.mode == 'gpu':
            diag_tmp = torch.mul(torch.eye(adj.shape[-1], device='cuda').repeat(adj.shape[0], 1),
                                 diag.repeat(1, adj.shape[-1]).view(adj.shape[0] * adj.shape[-1], -1))
        else:
            diag_tmp = torch.mul(torch.eye(adj.shape[-1]).repeat(adj.shape[0],1), diag.repeat(1,adj.shape[-1]).view(adj.shape[0]*adj.shape[-1],-1))
        adj_wo_diag = adj-diag_tmp.view(adj.shape[0], -1, adj.shape[-1])

        for gcn in self.gcns:
            hidden = gcn(hidden, adj, mask,epoch,pos)

        hidden = mask.unsqueeze(2) * hidden
        out_hidden = self.pool(hidden,mask)

        ###compute score
        nei_mask = adj_wo_diag.gt(0)
        # nei_mask_wo_dia = nei_mask.type(torch.FloatTensor) - torch.eye(adj.shape[-1]).unsqueeze(0)
        deg = torch.sum(adj,2)+1e-6


        deg_ten = nei_mask/(deg.repeat(1,adj.shape[-1]).view(adj.shape[0],-1,adj.shape[-1]))

        if self.proj_mode == 'sig':
            proj = self.sigmoid(self.proj(hidden))
        elif self.proj_mode == 'abs':
            proj = torch.abs(self.proj(hidden))
        # proj = torch.abs(self.proj(hidden))
        weight_norm = torch.norm(self.proj.weight,dim=1)
        score_tmp = (torch.matmul(deg_ten, proj)/weight_norm).squeeze(2)

        score_1 = torch.mul(score_tmp, mask)
        # score_1 = self.sigmoid(score)
        ss = torch.max(score_1,1,keepdim=True)[0]
        score = score_1/(torch.max(score_1,1,keepdim=True)[0]+1e-6)

        ###select nodes
        if self.sele_method == 1:
            top_index = self.find_top_k(score, nor_hops, k_max, k_list)

        #####
        if self.sele_method == 0:
            delta = self.find_delta(mask, score, nor_hops)
            gamma = torch.mul(score, delta)
            top_gamma, top_index = torch.topk(gamma, k_max, dim=1)


        sele_adj = adj.new_zeros(adj.shape[0],k_max, adj.shape[-1])
        sele_adj_2 = adj.new_zeros(adj.shape[0],k_max, adj.shape[-1])
        sele_adj_3 = adj.new_zeros(adj.shape[0],k_max, adj.shape[-1])
        sele_feat = adj.new_zeros(hidden.shape[0], k_max, hidden.shape[-1])
        adj_power2 = torch.matmul(adj,adj)
        adj_power3 = torch.matmul(adj_power2,adj)
        # adj_mul = adj+adj_power2+adj_power3
        for i, k in enumerate(k_list):
            idx = top_index[i][0:k]
            sele_feat[i][0:k] = hidden[i][idx,:]
            sele_adj[i][0:k] = adj[i][idx,:]
            sele_adj_2[i][0:k] = adj_power2[i][idx,:]
            sele_adj_3[i][0:k] = adj_power3[i][idx,:]
            new_mask[i][0:k] = 1.
            new_mask_assi[i][0:k] = 0.

        sele_nei_mask = torch.transpose(sele_adj, 2, 1).gt(0)
        sele_nei_mask_2 = torch.transpose(sele_adj_2, 2, 1).gt(0)
        sele_nei_mask_3 = torch.transpose(sele_adj_3, 2, 1).gt(0)

        #####compute B

        # mask_assign =  torch.mul(mask.unsqueeze(2).repeat(1,1, k_max), new_mask.unsqueeze(1).repeat(1,adj.shape[-1],1))
        B_tmp = torch.matmul(torch.matmul(hidden, self.W), torch.transpose(sele_feat, 2, 1))
        B_assi_g = B_tmp + new_mask_assi.unsqueeze(1)
        B_global = torch.nn.functional.softmax(B_assi_g, dim=2)*mask.unsqueeze(2)

        # sele_nei_mask_mul = sele_nei_mask + self.lamda1*sele_nei_mask_2 + self.lamda2*sele_nei_mask_3
        sele_nei_mask_mul = sele_nei_mask+ self.lamda1*sele_nei_mask_2
        sele_nei_mask_tmp = ((sele_nei_mask_mul.gt(0))*1-1)*100000000
        B_assi_l = torch.mul(B_tmp, sele_nei_mask_mul)
        B_assi_l = B_assi_l+sele_nei_mask_tmp
        B_local = torch.nn.functional.softmax(B_assi_l,dim=2)*mask.unsqueeze(2)

        #update X

        if self.aggre_mode == 'global':
            new_X = torch.matmul(torch.transpose(B_global,2,1), torch.mul(hidden, score.unsqueeze(2).repeat(1,1,hidden.shape[-1])))
        if self.aggre_mode == 'local':

            new_X = torch.matmul(torch.transpose(B_local,2,1),
                                 torch.mul(hidden, score.unsqueeze(2).repeat(1, 1, hidden.shape[-1])))

        ##update structure (adj)
        if self.struct_mode == 'global':
            new_adj = torch.matmul(torch.matmul(torch.transpose(B_global,2,1), adj), B_global)
        if self.struct_mode == 'local':
            mask_B = torch.mul(B_local,sele_nei_mask)
            new_adj = torch.matmul(torch.matmul(torch.transpose(B_local,2,1), adj), B_local)
        diag_elem = torch.pow(new_adj.sum(dim=2) + self.eps, -0.5)
        diag = new_adj.new_zeros(new_adj.shape)
        for i, x in enumerate(diag_elem):
            diag[i] = torch.diagflat(x)
        new_adj = torch.matmul(torch.matmul(diag, new_adj), diag)



        ##update distance
        new_nor_hops = adj.new_zeros(adj.shape[0], k_max, k_max)
        new_batch_graph = []
        visualize_tools = []
        if self.distance_mode == 0:
            for i, k in enumerate(k_list):
                idx = top_index[i][0:k]
                tmp = nor_hops[i][idx,:]
                tmp = tmp[:,idx]
                new_nor_hops[i, 0:k, 0:k] = tmp
        elif self.distance_mode == 1:
            new_nor_hops = torch.matmul(torch.matmul(torch.transpose(B_local,2,1), nor_hops), B_local)
        elif self.distance_mode == 2:
            new_nor_hops = torch.matmul(torch.matmul(torch.transpose(B_global, 2, 1), nor_hops), B_global)


        out = self.pool(new_X, new_mask)
        # if pos == 20:

        visualize_tools.append(top_index[0])
        visualize_tools.append(new_mask[0].sum())
        visualize_tools.append(new_adj[0])


        return out,out_hidden, new_X, new_adj, new_mask, new_nor_hops, visualize_tools
        # print(1)





class GCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, config,bn=0, add_self=0, normalize_embedding=0,
                 dropout=0.0, relu=0, bias=True):
        super(GCNBlock, self).__init__()
        dropout = 0.
        self.add_self = add_self
        self.dropout = dropout
        self.relu = relu
        self.bn = bn
        self.proj = nn.Linear(input_dim, output_dim)

        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        if self.bn:
            self.bn_layer = torch.nn.BatchNorm1d(output_dim)

        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim

        if torch.cuda.is_available() and config.mode == 'gpu':
            torch.cuda.set_device(config.gpu_device)
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
            torch.nn.init.xavier_normal_(self.weight)

            if bias:
                self.bias = nn.Parameter(torch.zeros(output_dim).cuda())
            else:
                self.bias = None
        else:
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
            torch.nn.init.xavier_normal_(self.weight)
            if bias:
                self.bias = nn.Parameter(torch.zeros(output_dim))
            else:
                self.bias = None

    def forward(self, x, adj, mask,epoch,pos):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)

        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        # if self.bn:
        #     index = mask.sum(dim=1).long().tolist()
        #     bn_tensor_bf = mask.new_zeros((sum(index), y.shape[2]))
        #     bn_tensor_af = mask.new_zeros(*y.shape)
        #     start_index = []
        #     ssum = 0
        #     for i in range(x.shape[0]):
        #         start_index.append(ssum)
        #         ssum += index[i]
        #     start_index.append(ssum)
        #     for i in range(x.shape[0]):
        #         bn_tensor_bf[start_index[i]:start_index[i + 1]] = y[i, 0:index[i]]
        #     bn_tensor_bf = self.bn_layer(bn_tensor_bf)
        #     for i in range(x.shape[0]):
        #         bn_tensor_af[i, 0:index[i]] = bn_tensor_bf[start_index[i]:start_index[i + 1]]
        #     y = bn_tensor_af
        # if self.dropout > 0.001:
        #     y = self.dropout_layer(y)
        if self.relu == 'relu':
            y = torch.nn.functional.relu(y)
        elif self.relu == 'lrelu':
            y = torch.nn.functional.leaky_relu(y, 0.1)
        return y

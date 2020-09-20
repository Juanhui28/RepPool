

import sys
import torch
import numpy as np
import random
from tqdm import tqdm
import os
import pickle
#import _pickle as cp  # python3 compatability
import networkx as nx
import argparse
import pandas as pd
import scipy.sparse as sp

# dataset = 'COLLAB'

cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-gpu', default='',type=str, help='gpu number')
cmd_opt.add_argument('-name', default='train', help='')
cmd_opt.add_argument('-print', type=int, default=0, help='')
cmd_opt.add_argument('-logdir', default='log', help='')
cmd_opt.add_argument('-savedir', default='save', help='')
cmd_opt.add_argument('-save', default=1, help='whether to save running metadata')
cmd_opt.add_argument('-save_feat', default=0, help='whether to save running metadata')
cmd_opt.add_argument('-init_from',type=str, default='', help='whether to save running metadata')
cmd_opt.add_argument('-save_freq', default=2, help='to save running metadata')
cmd_opt.add_argument('-sample', default='0,1,2', type=str,help='sample test minibatch for visulization')
cmd_opt.add_argument('-data', default='NCI1', help='data folder name')
cmd_opt.add_argument('-bsize', type=int, default=20, help='minibatch size')
cmd_opt.add_argument('-test_bsize', type=int, default=1, help='test minibatch size')
cmd_opt.add_argument('-seed', type=int, default=999, help='seed')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
cmd_opt.add_argument('-test_number', type=int, default=0, help='if specified, will overwrite -fold and use the last -test_number graphs as testing data')
cmd_opt.add_argument('-epochs', type=int, default=280, help='number of epochs')
cmd_opt.add_argument('-lr', type=float, default=1e-3, help='init learning_rate')
cmd_opt.add_argument('-decay', type=int, default=0, help='init learning_rate')
cmd_opt.add_argument('-patient', type=int, default=50, help='init learning_rate')
cmd_opt.add_argument('-clip', type=int, default=0, help='init learning_rate')
cmd_opt.add_argument('-max_grad_norm', type=float, default=1.5, help='init learning_rate')
cmd_opt.add_argument('-dropout', type=float, default=0.3, help='')
cmd_opt.add_argument('-printAUC', type=bool, default=False, help='whether to print AUC (for binary classification only)')

cmd_opt.add_argument('-aggre_mode', type=str, default='local', help='aggregation mode for feature (global or local)')
cmd_opt.add_argument('-struct_mode', type=str, default='local', help='update mode for structure (global or local)')
cmd_opt.add_argument('-sele_method', type=int, default='1', help='node selection method. 0 for selecting at the same time. 1 for selecting one by one.')
cmd_opt.add_argument('-lamda1', type=float, default=0.1, help='lamda1.')
cmd_opt.add_argument('-lamda2', type=float, default=0.1, help='lamda2.')
cmd_opt.add_argument('-gpu_device', type=int, default=1, help='gpu device number.')
cmd_opt.add_argument('-proj_mode', type=str, default='sig', help='normalization method for the projection. sig: sigmoid. abs: abs')
cmd_opt.add_argument('-distance_mode', type=int, default=0, help='update distance. 0 for taking original distance. 1 for update with local assign matrix. 2 for update with global matrix')
cmd_opt.add_argument('-max_node', type=int, default=1000, help='max nodes of graphs')


#classifier options:
cls_opt=cmd_opt.add_argument_group('classifier options')
cls_opt.add_argument('-model', type=str, default='agcn', help='model choice:gcn/agcn')
cls_opt.add_argument('-concat', type=int, default=0, help='model choice:gcn/agcn')
cls_opt.add_argument('-hidden_dim', type=int, default=64, help='hidden size k')
cls_opt.add_argument('-num_class', type=int, default=1000, help='classification number')
cls_opt.add_argument('-arch', type=int, default=2, help='layer number of agcn block')
cls_opt.add_argument('-num_layers', type=int, default=2, help='layer number of agcn block')
cls_opt.add_argument('-mlp_hidden', type=int, default=50, help='hidden size of mlp layers')
cls_opt.add_argument('-mlp_layers', type=int, default=2, help='layer number of mlp layers')
cls_opt.add_argument('-eps', type=float, default=1e-10, help='')

gcn_opt=cmd_opt.add_argument_group('gcn options')
gcn_opt.add_argument('-gcn_res', type=int, default=0, help='whether to use residual structure in gcn layers')
gcn_opt.add_argument('-gcn_norm', type=int, default=0, help='whether to normalize gcn layers')
gcn_opt.add_argument('-bn', type=int, default=0, help='whether to normalize gcn layers')
gcn_opt.add_argument('-relu', type=str, default='relu', help='whether to use relu')
gcn_opt.add_argument('-lastrelu', type=int, default=1, help='whether to use relu')
gcn_opt.add_argument('-gcn_layers', type=int, default=2, help='layer number in each agcn block')


#agcn options:
agcn_opt=cmd_opt.add_argument_group('agcn options')
agcn_opt.add_argument('-pool', type=str,default='max',help='agcn pool method: mean/max')
agcn_opt.add_argument('-percent', type=float,default=0.3,help='agcn node keep percent(=k/node_num)')




cmd_args = cmd_opt.parse_args()


class Hyper_Graph(object):
    def __init__(self, g, nor_hops, label=None, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        super().__init__()
        self.num_nodes = len(g)
        self.node_tags = self.__rerange_tags(node_tags, list(g.nodes()))  # rerangenodes index
        self.label = label
        self.g = g
        self.node_features = self.__rerange_fea(node_features, list(g.nodes()))  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree()).values())  # type(g.degree()) is dict
        self.adj = self.__preprocess_adj(nx.adjacency_matrix(g))  # torch.FloatTensor
        self.nor_hops = nor_hops
        self.node_idx = self.node2idx(list(g.nodes))

    def minimal_distance(self, g,node_idx):
        # g = graph.g
        # node_idx = graph.node_idx

        # hops = nx.all_pairs_shortest_path_length(g)
        hops = nx.all_pairs_dijkstra_path_length(g)
        hops = dict(hops)
        max_value = len(g)
        df = pd.DataFrame(hops).fillna(max_value)  ## for nodes without neighbors
        index = df.index.tolist()
        column = df.columns.tolist()
        vals = df.values.tolist()
        # ls.insert(0,df.columns.tolist())
        vals_tensor = torch.FloatTensor(vals)
        out_tmp = torch.zeros(len(g), len(g)).type(torch.FloatTensor)

        for i, value in enumerate(index):
            # out_tmp[value] = vals_tensor[i]
            out_tmp[node_idx[value]] = vals_tensor[i]
        out_tran = torch.transpose(out_tmp, 1, 0)

        dis = torch.zeros(len(g), len(g)).type(torch.FloatTensor)
        for i, value in enumerate(column):
            # dis[value] = out_tran[i]
            dis[node_idx[value]] = out_tran[i]
        dis = torch.transpose(dis, 1, 0)

        max_dis = torch.max(dis)
        tmp_dis = max_dis.unsqueeze(0).unsqueeze(1).repeat(dis.shape[0], dis.shape[1])

        nor_dis = torch.div(dis.type(torch.FloatTensor), tmp_dis)

        return dis, nor_dis


    def node2idx(self,node):
        idx = dict()
        for i in range(len(node)):
            idx[node[i]] = i
        return idx

    def __rerange_fea(self, node_features, node_list):
        if node_features == None or node_features == []:
            return node_features
        else:
            new_node_features = []
            for i in range(node_features.shape[0]):
                new_node_features.append(node_features[node_list[i]])

            new_node_features = np.vstack(new_node_features)
            return new_node_features

    def __rerange_tags(self, node_tags, node_list):
        if node_tags == None or node_tags == []:
            return node_tags
        else:
            new_node_tags = []
            if node_tags != []:
                for i in range(len(node_tags)):
                    new_node_tags.append(node_tags[node_list[i]])

            return new_node_tags

    def __sparse_to_tensor(self, adj):
        '''
            adj: sparse matrix in COOrdinate format
        '''
        assert sp.isspmatrix_coo(adj), 'not coo format sparse matrix'
        # adj = adj.tocoo()

        values = adj.data
        indices = np.vstack((adj.row, adj.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adj.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

    def __normalize_adj(self, sp_adj):
        adj = sp.coo_matrix(sp_adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        # return adj

    def __preprocess_adj(self, sp_adj):
        '''
            sp_adj: sparse matrix in Compressed Sparse Row format
        '''
        adj_normalized = self.__normalize_adj(sp_adj + sp.eye(sp_adj.shape[0]))


        return self.__sparse_to_tensor(adj_normalized)


def load_data():

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}
    count= 0
    print('data: /%s'% cmd_args.data)
    min_node = 10000000
    with open('data/%s/%sdistance.txt'%(cmd_args.data, cmd_args.data), 'rb') as fp:
        distance = pickle.load(fp)
    with open('data/%s/%s.txt' % (cmd_args.data,cmd_args.data), 'r') as f:
        n_g = int(f.readline().strip()) # number of graph
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]  # node number & graph label
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                # if int(row[1]) == 0:
                #     count += 1
                #     print('zeros neighbor:', count)
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])


            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            # assert len(g.edges()) * 2 == n_edges  (some graphs in COLLAB have self-loops, ignored here)
            assert len(g) == n
            hg = Hyper_Graph(g, distance[i], l, node_tags, node_features)
            g_list.append(hg)
            if len(g) < min_node:
                min_node = len(g)


    for g in g_list:
        g.label = label_dict[g.label]
    cmd_args.num_class = len(label_dict)
    cmd_args.feat_dim = len(feat_dict) # maximum node label (tag)
    if node_feature_flag == True:
        cmd_args.attr_dim = node_features.shape[1] # dim of node features (attributes)
    else:
        cmd_args.attr_dim = 0
    cmd_args.input_dim = cmd_args.feat_dim + cmd_args.attr_dim


    print('# classes: %d' % cmd_args.num_class)
    print('# maximum node tag: %d' % cmd_args.feat_dim)

    if cmd_args.test_number == 0:
        train_idxes = np.loadtxt('data/%s/10fold_idx/train_idx-%d.txt' % (cmd_args.data, cmd_args.fold), dtype=np.int32).tolist()
        test_idxes = np.loadtxt('data/%s/10fold_idx/test_idx-%d.txt' % (cmd_args.data, cmd_args.fold), dtype=np.int32).tolist()
        
        max_nodes = cmd_args.max_node
        train_graphs = []
        val_graphs = []
        for i in train_idxes:
            if max_nodes is not None and len(g_list[i].g) > max_nodes:
                continue
            train_graphs.append(g_list[i])
        for i in test_idxes:
            if max_nodes is not None and len(g_list[i].g) > max_nodes:
                continue
            val_graphs.append(g_list[i])
        return train_graphs, val_graphs
        
        #return [g_list[i] for i in train_idxes], [g_list[i] for i in test_idxes]
    else:
        return g_list[: n_g - cmd_args.test_number], g_list[n_g - cmd_args.test_number :]




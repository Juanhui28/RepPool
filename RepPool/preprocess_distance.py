

import sys
import torch
import numpy as np
import random
from tqdm import tqdm
import os
import pickle as cp
#import _pickle as cp  # python3 compatability
import networkx as nx
import argparse
import pandas as pd
import scipy.sparse as sp
import pickle

# dataset = 'COLLAB'

# cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
# cmd_opt.add_argument('-num_class', type=int, default=1000, help='classification number')
# cmd_opt.add_argument('-data', default='NCI1', help='data folder name')
#
# cmd_args = cmd_opt.parse_args()


class Hyper_Graph(object):
    def __init__(self, g, label=None, node_tags=None, node_features=None):
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


        self.node_idx = self.node2idx(list(g.nodes))
        _,self.nor_hops = self.minimal_distance(g, self.node_idx)

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


def load_data(data):

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}
    count= 0
    print('data: /%s'% data)
    min_node = 10000000
    nor_hops = []
    with open('data/%s/%s.txt' % (data, data), 'r') as f:
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
            hg = Hyper_Graph(g, l, node_tags, node_features)
            g_list.append(hg)
            if len(g) < min_node:
                min_node = len(g)
            nor_hops.append(hg.nor_hops)


    # for g in g_list:
    # #     g.label = label_dict[g.label]
    #       nor_hops.append(g.nor_hops)
    with open('data/'+data+'/'+data+'distance.txt','wb') as fp:
        pickle.dump(nor_hops, fp)

data = 'NCI109'
load_data(data)


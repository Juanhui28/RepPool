
import torch
import torch.nn as nn
import math
import pandas as pd
import networkx as nx
import random
import sys
import os
import time
from gcn import *
import numpy as np
import torch.optim as optim
from sklearn import metrics
from mlp_dropout import MLPClassifier
from util import cmd_args, load_data

args = cmd_args
if args.init_from!='':
    tmp=args.init_from
    state_dict=torch.load(args.init_from)
    args=state_dict['args']
    args.init_from=tmp

save_path=os.path.join(args.savedir,args.data,str(args.percent),str(args.lr),str(args.fold))
if os.path.exists(save_path):
    os.system('rm -rf '+save_path)
os.makedirs(save_path)
# max_value = sys.maxsize
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.agcns = nn.ModuleList()
        x_size = args.input_dim
        self.num_layers = args.num_layers

        if args.arch == 1:
            assert args.gcn_layers % self.num_layers == 0
            gcn_layer_list = [args.gcn_layers // self.num_layers] * self.num_layers
        elif args.arch == 2:
            gcn_layer_list = [args.gcn_layers] + [1] * (self.num_layers - 1)

        for i in range(args.num_layers):
            self.agcns.append(AGCNBlock(args, x_size, args.hidden_dim, gcn_layer_list[i], args.dropout, args.relu))
            x_size = self.agcns[-1].pass_dim

        self.mlps = nn.ModuleList()
        self.mlps = MLPClassifier(input_size=args.hidden_dim*args.num_layers*2, hidden_size=args.mlp_hidden,
                                  num_class=args.num_class,  dropout=args.dropout)




    def PrepareFeatureLabel(self, batch_graph):
        batch_size = len(batch_graph)
        labels = torch.LongTensor(batch_size)
        max_node_num = 0


        for i in range(batch_size):
            labels[i] = batch_graph[i].label
            max_node_num = max(max_node_num, batch_graph[i].num_nodes)

            # print('tags:',batch_graph[i].node_tags)
        masks = torch.zeros(batch_size, max_node_num)
        mask_assi = torch.ones(batch_size,max_node_num)*(-10000.)
        adjs = torch.zeros(batch_size, max_node_num, max_node_num)
        degs = torch.zeros(batch_size, max_node_num)
        distances = torch.zeros(batch_size, max_node_num, max_node_num)
        nor_distances = torch.zeros(batch_size, max_node_num, max_node_num)

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            batch_node_tag = torch.zeros(batch_size, max_node_num, args.feat_dim)
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            batch_node_feat = torch.zeros(batch_size, max_node_num, args.attr_dim)
        else:
            node_feat_flag = False

        for i in range(batch_size):
            cur_node_num = batch_graph[i].num_nodes
            # tmp_deg = batch_graph[i].degs

            if node_tag_flag == True:
                tmp_tag_idx = torch.LongTensor(batch_graph[i].node_tags).view(-1, 1)
                tmp_node_tag = torch.zeros(cur_node_num, args.feat_dim)
                tmp_node_tag.scatter_(1, tmp_tag_idx, 1)
                batch_node_tag[i, 0:cur_node_num] = tmp_node_tag
            # node attribute feature
            if node_feat_flag == True:
                tmp_node_fea = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
                batch_node_feat[i, 0:cur_node_num] = tmp_node_fea

            #distance
            # distances[i, 0:cur_node_num, 0:cur_node_num], nor_distances[i, 0:cur_node_num, 0:cur_node_num]= batch_graph[i].minimal_distance(batch_graph[i].g, batch_graph[i].node_idx)
            nor_distances[i, 0:cur_node_num, 0:cur_node_num] = batch_graph[i].nor_hops

            #degrees
            degs[i, 0:cur_node_num] = torch.LongTensor(batch_graph[i].degs)

            # adjs
            adjs[i, 0:cur_node_num, 0:cur_node_num] = batch_graph[i].adj

            # masks
            masks[i, 0:cur_node_num] = 1


            # cobime the two kinds of node feature
        if node_feat_flag == True:
            node_feat = batch_node_feat.clone()

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([batch_node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = batch_node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(batch_size, max_node_num, 1)  # use all-one vector as node features

        if args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()
            adjs = adjs.cuda()
            masks = masks.cuda()

            nor_distances = nor_distances.cuda()

        return node_feat, labels, adjs, masks, degs, nor_distances

    def forward(self, batch_graph, epoch, pos, is_print=False):
        node_feat, labels, adjs, masks, degs, nor_hops = self.PrepareFeatureLabel(batch_graph)
        k_max = int(math.ceil(args.percent * adjs.shape[-1]))
        k_list = [int(math.ceil(args.percent * x)) for x in masks.sum(dim=1).tolist()]
        X = node_feat
        visualize_tools = []
        visualize_tools1 = [labels.cpu()]

        out_all = []
        for i in range(self.num_layers):

            out,out_hidden, X, adjs, masks, nor_hops, visualize_tool = self.agcns[i](X, adjs, masks, nor_hops,i,epoch,pos,is_print=is_print)
            visualize_tools.append(visualize_tool)
            out_all.append(out_hidden)
            out_all.append(out)

            if args.save_feat and not self.training:
                visualize_tools1.append([out.cpu(),X.cpu(),masks.cpu()])


        output = torch.cat(out_all,dim=1)
        logits, loss, acc = self.mlps(output, labels)


        return logits, loss, acc, visualize_tools,visualize_tools1


def loop_dataset(g_list, classifier, sample_idxes, epoch,optimizer=None, bsize=50):


    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    all_targets = []
    all_scores = []
    visual_pos = [int(x) for x in args.sample.strip().split(',')]

    n_samples = 0
    total_loss = []
    vis1 = []
    for pos in range(total_iters):
        visualize_tools = []

        selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]
        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets

        logits, loss, acc,visualize_tools,visualize_tools1 = classifier(batch_graph, epoch,pos,is_print=(pos in visual_pos))
        vis1.append(visualize_tools1)
        all_scores.append(logits[:, 1].detach())  # for binary classification
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(),args.max_grad_norm)
            optimizer.step()

        if args.save and (pos in visual_pos) and (not classifier.training)  and epoch%args.save_freq==0 :
        # if args.save and classifier.training:
            visualize_tools = list(zip(*visualize_tools))
            # visualize_tools = [x.detach().cpu().numpy() for x in visualize_tools]
            visualize_tools = [[x.detach().cpu().numpy() for x in y] for y in visualize_tools]

            np.save(os.path.join(save_path, 'Mysample%03d_epoch%03d.npy' % (pos, epoch)),
               [batch_graph[0].g, batch_graph[0].node_tags] + visualize_tools)

        loss = loss.data.cpu().numpy()
        total_loss.append(np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)

    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()

    # np.savetxt('test_scores.txt', all_scores)  # output test predictions

    all_targets = np.array(all_targets)
    fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    avg_loss = np.concatenate((avg_loss, [auc]))

    return avg_loss, vis1



if __name__ == '__main__':
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_graphs, test_graphs = load_data()
    classifier = Classifier()
    if args.mode == 'gpu' and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_device)
        classifier = classifier.cuda()
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)

    train_idxes = list(range(len(train_graphs)))

    best_acc = float('-inf')
    best_auc = float('-inf')

    if args.init_from!='':
        classifier.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optim_state_dict'])
        start_epoch=state_dict['epoch']
        best_acc=state_dict['best_acc']

        dummy_idxes=list(range(len(train_graphs)))
        for _ in range(start_epoch):
            random.shuffle(dummy_idxes)

    for epoch in range( args.epochs): #args.epochs

        # random.Random(0).shuffle(train_idxes)
        random.shuffle(train_idxes)
        # print(train_idxes)
        start_time = time.time()
        classifier.train()
        avg_loss,vis = loop_dataset(train_graphs, classifier, train_idxes, epoch, optimizer=optimizer, bsize=args.bsize)

        print('=====>average training of epoch %d: loss %.5f acc %.5f auc %.5f' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2]))
        #        log_value('train acc',avg_loss[1],epoch)

        classifier.eval()

        test_loss, vis = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))), epoch,
                                                      bsize=args.test_bsize)
        if best_acc < test_loss[1]:
            best_acc = test_loss[1]
            best_epoch = epoch
            torch.save({'model_state_dict': classifier.state_dict(),
                        'optim_state_dict': optimizer.state_dict(),
                        'args': args,
                        'epoch': epoch,
                        'best_acc': best_acc},
                       os.path.join(save_path, 'best_model.pth'))
            torch.save(vis, os.path.join(save_path, 'best_feature.pth'))
        if best_auc < test_loss[2]:
            best_auc = test_loss[2]
        print('=====>average testing of epoch %d: loss %.5f acc %.5f auc %.5f best_acc %.5f best_auc %.5f time: %.0fs' % (epoch, test_loss[0], test_loss[1], test_loss[2],best_acc, best_auc, time.time()-start_time))
        print('\n')

    with open('save/acc_result_%s_%s_%s_%s_%s_%s.txt' %(args.data,str(args.lr),str(args.hidden_dim),str(args.percent),str(args.num_layers),str(args.gcn_layers)), 'a+') as f:
        # f.write(str(test_loss[1]) + '\n')
        f.write(str(best_acc) + '\n')

import numpy as np
import networkx as nx
import  matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import pylab
# def visualize(orig_graph, att_list, index_list, A, mask_list):
def my_visualize(g, node_tags, index_list, mask_list, A):
    '''
    		orig_graph: an instance in graph list
    		att_list: a list of att tensor
    		index_list: a list of index tensor that indicates the rank of nodes
    		A: a list of adjacent matrix after pooling
    		mask_list: an longtensor
    	'''
    path = './visual/'
    color = ['red', 'green', 'blue']
    neighbour_color = 'yellow'
    origin_color = 'black'

    # g = orig_graph.g
    ag = nx.nx_agraph.to_agraph(g)
    # add node tag
    for i, tag in enumerate(node_tags):
        node = ag.get_node(list(g.nodes())[i])

        node.attr['label'] = int(node)
        # node.attr['style'] = 'circle'

    ag.layout()
    file_name = path + 'origin.png'
    ag.draw(file_name)
    # nx.draw(g)
    img = plt.imread(file_name)
    # plt.imshow(img)
    plt.show()
    # pylab.show()

    for i,_ in enumerate(mask_list):
        mask = int(mask_list[i])
        # for j in range(mask):
        #     tmp = g.nodes()
        #     cur_idx = list(g.nodes())[index_list[i][j]]
        #     node = ag.get_node(cur_idx)
        #     node.attr['color'] = color[i]
        #     neighbour_list = list(g.adj[cur_idx])
        #     for k, idx_t in enumerate(neighbour_list):
        #         # idx = list(g.nodes())[idx_t]
        #         idx = idx_t
        #         node = ag.get_node(idx)
        #         node.attr['color'] = neighbour_color
        #         edge = ag.get_edge(cur_idx, idx)
        #         edge.attr['color'] = neighbour_color
        #     ag.layout()
        #     # file_name = path + 'att_' + str(i) + '_' + str(j) + '.png'
        #     # ag.draw(file_name)
        #
        #     # restore the change
        #     node = ag.get_node(cur_idx)
        #     node.attr['color'] = origin_color
        #     for k, idx_t in enumerate(neighbour_list):
        #         # idx = list(g.nodes())[idx_t]
        #         idx = idx_t
        #         node = ag.get_node(idx)
        #         node.attr['color'] = origin_color
        #         edge = ag.get_edge(cur_idx, idx)
        #         edge.attr['color'] = origin_color

        for j in range(mask):
            cur_idx = list(g.nodes())[index_list[i][j]]
            print(i, cur_idx)

            node = ag.get_node(cur_idx)

            node.attr['color'] = color[i]

            # neighbour_list = list(g.adj[cur_idx])
        # for k, idx_t in enumerate(neighbour_list):
        # 	idx = list(g.nodes())[idx_t]
        # 	node = ag.get_node(idx)
        # 	node.attr['color'] = neighbour_color
        # 	edge = ag.get_edge(cur_idx, idx)
        # 	edge.attr['color'] = neighbour_color
        ag.layout()
        file_name = path + 'after_att_' + str(i) + '.png'
        ag.draw(file_name)
        # img = plt.imread(file_name)
        # plt.imshow(img)
        # plt.show()

        # new g
        adj = A[i]
        new_g = nx.Graph()
        for j in range(mask):
            cur_idx = list(g.nodes())[index_list[i][j]]
            new_g.add_node(cur_idx)
            for k in range(mask):
                if adj[j][k] > 0:
                    new_g.add_edge(cur_idx, list(g.nodes())[index_list[i][k]])
        new_ag = nx.nx_agraph.to_agraph(new_g)
        for j in range(mask):
            # cur_idx = index_list[i][j]
            cur_idx = list(g.nodes())[index_list[i][j]]
            node = new_ag.get_node(cur_idx)
            node.attr['label'] = ag.get_node(list(g.nodes())[index_list[i][j]]).attr['label']
        g = new_g
        ag = new_ag
        ag.layout()
        file_name = path + 'after_pool_' + str(i) + '.png'
        ag.draw(file_name)

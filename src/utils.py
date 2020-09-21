import json
import pdb
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import os.path as osp
from os import listdir, rmdir
from shutil import move
from torch_geometric.utils import remove_self_loops, add_self_loops


def self_loop_augment(num_nodes, adj):
    adj, _ = remove_self_loops(adj)
    adj, _ = add_self_loops(adj, num_nodes=num_nodes)
    return adj


def move_to_root(root):
    for n in listdir(root):
        p = osp.join(root, n)
        if osp.isdir(p):
            for fn in listdir(p):
                try:
                    move(osp.join(root, n, fn), osp.join(root, fn))
                except IOError:
                    print("cannot move file ", fn)
            rmdir(p)


# save a list of graphs
def save_graph_list(graphs, filename, clean=False, has_par=False, nodes_par1_list=None, nodes_par2_list=None):
    with open(filename, "wb") as f:
        graphs_info = []
        for i, gr in enumerate(graphs):
            if clean:
                gr = max(nx.connected_component_subgraphs(gr), key=len)
            if has_par:
                graphs_info.append([gr.nodes(), gr.edges(), nodes_par1_list[i], nodes_par2_list[i]])
            else:
                graphs_info.append([gr.nodes(), gr.edges()])
        pickle.dump(graphs_info, f)


def load_graph_list(filename, has_par=False):
    with open(filename, "rb") as f:
        graphs = []
        if has_par:
            nodes_par1_list = []
            nodes_par2_list = []
        graphs_info = pickle.load(f)
        for graph_info in graphs_info:
            g = nx.Graph()
            g.add_nodes_from(graph_info[0])
            g.add_edges_from(graph_info[1])
            graphs.append(g)
            if has_par:
                nodes_par1_list.append(graph_info[2])
                nodes_par2_list.append(graph_info[3])
    if has_par:
        return graphs, nodes_par1_list, nodes_par2_list
    else:
        return graphs


# draw a list of graphs [G]
def draw_graph_list(graphs, row, col, filename='figures/test', layout='spring', is_single=False, k=1, node_size=55,
                    alpha=1, width=1.3):
    # # draw graph view
    # from pylab import rcParams
    # rcParams['figure.figsize'] = 12,3
    if len(graphs) > row * col:
        graphs = graphs[:row * col]
    plt.switch_backend('agg')
    for i, G in enumerate(graphs):
        plt.subplot(row, col, i + 1)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1,
                            wspace=0, hspace=0)
        plt.axis("off")
        if layout == 'spring':
            pos = nx.spring_layout(G, k=k / np.sqrt(G.number_of_nodes()), iterations=20)  # default 100
            # pos = nx.spring_layout(G)

        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        # # nx.draw_networkx(G, with_labels=True, node_size=2, width=0.15, font_size = 1.5, node_color=colors,pos=pos)
        # nx.draw_networkx(G, with_labels=False, node_size=1.5, width=0.2, font_size = 1.5, linewidths=0.2,
        # node_color = 'k',pos=pos,alpha=0.2)

        if is_single:
            # node_size default 60, edge_width default 1.5
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='#336699', alpha=1, linewidths=0,
                                   font_size=0)
            nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
        else:
            nx.draw_networkx_nodes(G, pos, node_size=1.5, node_color='#336699', alpha=1, linewidths=0.2, font_size=1.5)
            nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.2)

    plt.tight_layout()
    plt.savefig(filename + '.png', dpi=600)
    plt.close()

def make_checkpoint(path, epoch, model, optimizer, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

if __name__ == '__main__':
    import os
    configs_file = os.path.join(os.getcwd(), 'best_configs.json')
    with open(configs_file, 'r') as f:
        configs = json.load(f)
    pdb.set_trace()
    f_name = 'GCN_3_32_preTrue_dropFalse_yield1_08000.dat'
    grs = load_graph_list('graphs/' + f_name)
    graph = grs[0]
    pdb.set_trace()
    draw_graph_list(grs, row=4, col=4, filename='fig/' + f_name)

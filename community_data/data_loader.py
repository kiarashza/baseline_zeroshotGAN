import networkx as nx
import scipy
import numpy as np

import numpy
from operator import itemgetter
import random


def conditional_n_community(c_sizes, p_inter=0.1, p_intera=0.4):
    A_ = [] #a list of type_i adj
    D_ = [] #descriotion of type_i linl
    W_ = [] # the lambda in stochastic model
    graphs = [nx.gnp_random_graph(c_sizes[i], p_intera, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = list(nx.connected_components(G))



    Z = np.zeros((sum(c_sizes),len(c_sizes)),dtype=np.float32)
    for i in range(len(communities)):
        G_i = nx.Graph()
        G_i.add_nodes_from([i for i in range(sum(c_sizes))])
        G_i.add_edges_from(G.edges(communities[i]))
        A_.append(G_i)
        source = [0 for i in range(len(communities))]
        source[i] = 1
        feature =  source
        feature.extend(source)
        feature.append(p_intera)
        D_.append(feature)
        w_i = np.zeros((len(c_sizes),len(c_sizes)))
        w_i[i,i]=p_intera
        W_.append(w_i)
        Z[:,i] = (nx.to_scipy_sparse_matrix(G_i).sum(0)>0)*(1)

    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1)
        for j in range(i+1, len(communities)):
            G_i = nx.Graph()
            G_i.add_nodes_from([i for i in range(sum(c_sizes))])
            subG2 = communities[j]
            nodes2 = list(subG2)
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        G_i.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
                G_i.add_edge(nodes1[0], nodes2[0])
            source =  [0 for i in range(len(communities))]
            dist =  [0 for i in range(len(communities))]
            source[i] = 1
            dist[j] = 1
            feature = source
            feature.extend(dist)
            feature.append(p_inter)
            D_.append(feature)
            A_.append(G_i)
            w_i = np.zeros((len(c_sizes), len(c_sizes)))
            w_i[i,j] = p_inter
            w_i[j, i] = p_inter
            W_.append(w_i)

    print('connected comp: ', len(list(nx.connected_components(G))))
    A = [nx.to_scipy_sparse_matrix(G_i) for G_i in A_]
    D = [np.array(D_i) for D_i in D_]
    return A, D, scipy.sparse.lil_matrix(scipy.sparse.identity(A[0].shape[0])), Z, W_

if __name__ == '__main__':
    conditional_n_community([50,50,50])
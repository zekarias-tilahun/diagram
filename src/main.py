from sklearn import preprocessing

import networkx as nx
import numpy as np

import argparse

import model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n-file', default='../data/cora/graph.txt', type=str, help='Path to a graph file')
    parser.add_argument('--f-file', default='../data/cora/feature.txt', type=str, help='Path to node features file')
    parser.add_argument('--nm-file', type=str, default='../data/cora/nm_model.npz',
                        help="A path to save the learned embeddings of Diagram's node model variant")
    parser.add_argument('--em-file', type=str, default='../data/cora/em_model.npz',
                        help="A path to save the learned embeddings of Diagram's edge model variant")
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    return parser.parse_args()


def load_data(net_path, feature_path, normalize=False, rate=None):
    print('Loading data graph from {}'.format(net_path))
    graph = nx.read_edgelist(net_path, nodetype=int, create_using=nx.DiGraph())
    graph_mat = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()), dtype=int)
    
    print('Loading data graph from {}'.format(feature_path))
    feature_mat = np.loadtxt(feature_path)
    
    num_nodes = min(graph_mat.shape[0], feature_mat.shape[0])
    graph_mat = graph_mat[:num_nodes]
    feature_mat = preprocessing.normalize(feature_mat[:num_nodes],
                                          norm='l2', axis=1) if normalize else feature_mat[:num_nodes]
    print('\tAdjacency matrix shape: {}'.format(graph_mat.shape))
    print('\tFeature matrix shape: {}'.format(feature_mat.shape))
    print('\tNumber of edges: {}'.format(graph.number_of_edges()))
    return graph, graph_mat,  feature_mat


class Options:

    def __init__(self, number_of_nodes, number_of_features, learning_rate=0.0001, rate=0.2, layers=None,
                 d=128, unify_method='sum', beta=10, samples=20, epochs=10, batch_size=32, seed=0):
        self.number_of_nodes = number_of_nodes
        self.number_of_features = number_of_features
        self.layers = [512, 256] if layers is None else layers
        self.d = d
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.rate = rate
        self.seed = seed
        self.beta = beta
        self.samples = samples
        self.unify_method = unify_method

    def __str__(self):
        s = u'Number of nodes: {}, Number of features: {}, Layers:{}, Size: {}, Epochs: {}, Batch size: {}'.format(
            self.number_of_nodes, self.number_of_features, ' '.join(str(l) for l in self.layers),
            self.d, self.epochs, self.batch_size)
        return s


def main():
    args = parse_args()
    graph, adj_mat, features = load_data(net_path=args.n_file, feature_path=args.f_file)
    options = Options(number_of_nodes=adj_mat.shape[0], number_of_features=features.shape[1])

    node_model = model.NodeModel(options=options, adj=adj_mat, features=features)
    node_model.train(epochs=10)
    node_model.save(args.nm_file)

    edge_model = model.EdgeModel(options=options, adj=adj_mat, features=features,
                                 transferred_weights=node_model.get_learned_weights())
    edge_model.train(epochs=1)
    edge_model.save(args.em_file)

    
if __name__ == '__main__':
    main()
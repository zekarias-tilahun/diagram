# Diagram
A simple implementation of the paper "Which way? Direction-Aware Attributed Graph Embedding"

### Requirements

  - tensorflow 1.12.0+
  - numpy 1.16.5+
  - networkx 2.2+
  - scikit-learn 0.20.3+


#### Example usage

```sh
$ cd src
$ python main.py
```

#### Or

A Sample execution along with the node classification evaluation is given in the jupyter  notebook file under the src directory.

### Input arguments

`--n-file:` Path to a directed graph file, in an edge list format. Default is '../data/cora/graph.txt'

`--f-file:` Path to node features file.

`--nm-file:` A path to save the learned embedding of Diagram's node model variant. Default is '../data/cora/nm_model.npz'.

`--em-file:` A path to save the learned embeddings of Diagram's edge model variant. Default is '../data/cora/em_model.npz'

`--lr:` Learning rate. Default is 0.0001


Citing
------

```
@inproceedings{GEMKefato2019,
  booktitle="GEM: Graph Embedding and Mining workshop",
	author = {Kefato, Zekarias T. and Sheikh, Nasrullah and Montresor, Alberto},
	title = "Which way? Direction-Aware Attributed Graph Embedding",
	year = 2019,
}
```
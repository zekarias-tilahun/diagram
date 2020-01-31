# Diagram
A simple implementation of the paper "Which way? Direction-Aware Attributed Graph Embedding" (https://arxiv.org/abs/2001.11297) presented at the GEM: Graph Embedding and Mining workshop collocated with ECML-PKDD 2019 conference.

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
@inproceedings{kefato2020way,
    title={Which way? Direction-Aware Attributed Graph Embedding},
    author={Zekarias T. Kefato and Nasrullah Sheikh and Alberto Montresor},
    year={2020},
    eprint={2001.11297},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
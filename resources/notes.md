# Graph Convolutional Networks

## Why Graphs?

Across domains and industries, much important actual data is represented in the form of graphs: [Social networks, protein-interaction networks, knowledge graphs, the World Wide Web, etc.](Kipf, Graph Convolutional Networks).

<!-- Understanding the connections and relationships between our data help us to make better predictions ? -->

## Existing Approaches

Previous to around 2014, 2015, most approaches were dominated by *kernel-based methods*, *graph-based regularization techniques*, etc. while little work had been done to generalize neural networks to graph-structured data. (Kipf)

My notes:

  - Regularization rather than focusing on generalizing to be able to actually predict on arbitrary graph-structured data
  - Look up kernel-based methods - what are they?
  - Review literature in "Recent Literature" from Kipf on existing methods of generalizing RNNs or CNNs to graphs - papers

### Drawbacks

## What are GCN's?

*The following is taken directly from Kipf, paraphrase later*

1. The common "universal" architecture shared between most "graph neural network models"

2. These are what is described by Kipf as "Graph Convolutional Networks"

  - "Convolutional, because filter parameters are shared over all locations in the graph - or a subset thereof (cites Duvenaud et. al)"
  - Goal of these models is to learn a function of signals/features on a graph *G = (V, E)* which takes as input:
    - A feature description *xi* for every node *i*; summarized in a *N x D* feature matrix *X* (*N*: number of nodes, *D*: number of input features)
    - A representative description of the graph structure in matrix form; typically in the form of an adjacency matrix *A* (or some function thereof)

    And to produce a node-level output *Z* (an *N x F* feature matrix, where *F* is the number of output features per node). Graph-level outputs can be modeled by introducing some form of pooling operation.

See Kipf for further mathematical breakdown of this problem.

*Getting the Intuition of Graph Neural Networks*:

- Spektral, a Python library for Graph Neural Networks
  - Based on Keras and Tensorflow
- Aside: *Complete Graph* Is a graph where all nodes are connected to all other nodes.
- To implement GNNs with Spektral, we represent the graphs as matrices. The several matrices we need are:
  1. Adjacency matrix
  2. Node Attributes Matrix
    - Represents the features or attributes of each node
    - Size of this matrix is *N* x *F* where *N* is the number of nodes, and *F* the number of features
  3. Edge Attributes Matrix
    - Sometimes edges can have attributes, too (i.e: weights (where here I reference weights in terms of "weighted" graph, rather than "weights" as we know them from NN's) representing "strength" of relationships, etc.)
    - If the size of edge attributes is *S* and the number of edges available is *n_edges* then the shape would be *n_edges x S*
- **Single Mode vs. Batch Mode Representation**
  - In the molecular example in this article, each molecule is its own graph. As a result, we have many different graphs to train on.
    - this is referred to as *Batch Mode*
  - In the document citation example, we only have one large graph.
    - This is referred to as *Single Mode*


**Graph Neural Networks vs. Convolutional Neural Networks**:

- Effectively, CNN's view images (for example) implicitly as a graph, viewing pixels as:
  - the pixel "value" (brightness, strength, darkness to us)
  - the distance between other pixels

However, where CNN's reach their limitation is non-Euclidean data, data without a regular structure, or which is more than 2-dimensional. This is where GNN's come in - for things like:

  - social media Networks
  - Three-dimensional images
  - molecular structure
  - biological networks


*Graph Convolutional Networks for Node Classification*:

- GCN's - one of the basic Graph Neural Network variants
  - "Convolution" refers to "multiplying the input neurons with a set of weights that are commonly known as 'filters' or 'kernels'"
    - the features act as a 'sliding window' across the whole image and enable CNNs to learn features from neighboring cells.
  - GCN's perform similar operations where the model learns features by inspecting neighboring nodes.

- Two distinct types of Graph Convolutional Networks:

  1. Spatial Graph Convolutional Networks
  2. Spectral Graph Convolutional Networks
    - Fast Approximation Spectral-based GCN

- NN forward propagation quick recap:

  - In order to propagate the features representation in a neural network to the next layer, we perform the following:

    H^[i+1] = sigma(W^[i]H^[i] + b^[i])

    Where *H^[i+1]* ('H for *i + 1*') is the feature representation at layer *i + 1*
    Sigma represents our *activation function*
    *W^[i]* ('W for i') represents our weights at layer *i*
    *H^[i]* ('H for i') represents our feature representation at layer i
    *b^[i]* ('b for i') represents our bias at layer i

    So, we're propagating our weights x our feature representation + our bias to achieve our *i + 1* (our *next*) layer.

    This is basically equivalent to y = mx + b in linear regression where:

    *m* is equivalent to the weights
    *x* is the input features
    *b* is the bias

    What distinguishes our above equation from the linear regression equation is that neural networks apply a non-linear "activation function" in order to represent the non-linear features in latent dimension

So how does this equation differ in graph convolutional networks?

We're going to take adjacency matrix *A* and insert it into our above equation. By inserting *A*, we enable the model to learn the feature representations based on nodes' connectivity. For the sake of simplicity, you'll see in the following example bias *b* is omitted.

    By adding the adjacency matrix *A*, the forward pass equation will now be as follows:

    H^[i+1] = sigma(W^[i]H^[i]A*)

    Where `A*` represents normalized *A*. We'll talk about why we need to normalize *A* in a moment.

 

### Why GCN's?

### Drawbacks

Limitation of the above model (Kipf):

  1. Multiplication with *A* means that, for every node, we sum up all the feature vectors of all neighboring nodes but not the node itself (unless there are self-loops in the graph). We can "fix" this by enforing self-loops in the graph: by adding the identity matrix to *A*

  2. *A* is typically not normalized, and the multiplication with *A* will completely change the scale of the feature vectors (we can understand that by looking at the eigenvalues of *A*). Normalizing *A* such that all rows sum to one, i.e. *D^-1A* where *D* is the diagonal node degree matrix, gets rid of this problem. Multiplying with *D^-1A* now corresponds to taking the average of neighboring node features.

    - Review portion about symmetric normalization in Kipf



## How to Develop GCN's

#### Kipf & Welling

Given the propogation rule established by Kipf & Welling and discussed in the Kipf article, see example in article of 3-layer GCN

  1. Input: adjacency matrix of the graph and *X=I* (i.e: the identity matrix, prior to having any node features)
  2. The three-layer GCN now performs three propogation steps during the forward pass and effectively convolves the 3rd order neighborhood of every node (all nodes up to three "hops" away)

    - The model produces an embedding of the nodes that closely resemble the community structure (see [Zachary's karate club network](https://en.wikipedia.org/wiki/Zachary%27s_karate_club))
    - Note that this is before any training updates - with the weights being completely random

    - **Compare to DeepWalk, which achieves a similar embedding with a complicated unsupervised training procedure**

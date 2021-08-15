# _Propositional Satisfiability Problem_ (SAT) goes neural and deep

We'd like to use graph neural network to solve the SAT (more general _Constraint Satisfaction Problem_) and this repo is the PyTorch implementation of paper [___Transformers Satisfy___](https://openreview.net/pdf?id=Gj9aQfQEHRS) and [___Transformer-based Machine Learning for Fast SAT Solvers and Logic Synthesis___](https://arxiv.org/abs/2107.07116)

## General questions:

[How to convert a __CSP__ to _k_-__SAT__ ](https://cs.stackexchange.com/questions/23157/transformation-of-constraint-satisfaction-to-sat): For _SAT_, there are _n_ binary variables and _m_ constraints. Each constraint is associated with a _k_-tuple of (distinct) variables, for some _k_ depending on the constraint, along with a subset of <img src="https://render.githubusercontent.com/render/math?math={0,1}"> _k_ which is the allowed assignments for the _k_-tuple of variables. For example, graph coloring, or rather, whether a given graph _G_ can be _x_-colored, can be viewed as a CSP. If there are _n_ vertices (we identify the vertex set with <img src="https://render.githubusercontent.com/render/math?math={1,\dots,n}"> then we have _n_ groups <img src="https://render.githubusercontent.com/render/math?math=x_1,\dots,x_n"> of <img src="https://render.githubusercontent.com/render/math?math=\big \lceil \log_2 x \big \rceil"> variables. For each <img src="https://render.githubusercontent.com/render/math?math=i \in \{1,\dots, n\}"> there is a constraint on each group <img src="https://render.githubusercontent.com/render/math?math=x_i"> stating that the assignment for <img src="https://render.githubusercontent.com/render/math?math=x_i"> is the binary encoding of a number in the range <img src="https://render.githubusercontent.com/render/math?math={0,\dots,x - 1}">. For any two connected vertices <img src="https://render.githubusercontent.com/render/math?math=(i, j)"> there is a constraint on both groups <img src="https://render.githubusercontent.com/render/math?math=x_i, x_j"> stating that <img src="https://render.githubusercontent.com/render/math?math=x_i \neq x_j">. This CSP is satisfiable iff _G_ is _x_-colorable. In order to convert such a binary CSP to a SAT instance, we replace each constraint with the corresponding CNF. Continuing the example above, if <img src="https://render.githubusercontent.com/render/math?math=x=2"> then for any <img src="https://render.githubusercontent.com/render/math?math=i,j"> there is a constraint <img src="https://render.githubusercontent.com/render/math?math=x_i \neq x_j"> which we realize as <img src="https://render.githubusercontent.com/render/math?math=(x_i \vee \neg x_j) \wedge (\neg x_i \vee x_j)">. The resulting CNF is satisfiable iff the CSP is. You can use the standard reduction to convert this CNF to a 3CNF if you wish. We can also handle CSPs which are not binary, i.e., in which the variables are not binary but rather have some finite domain. The idea is similar to how I described coloring as a CSP above, and left to the reader.


## References:
+ Transformers are graph neural networks [[link](https://docs.dgl.ai/en/latest/tutorials/models/4_old_wines/7_transformer.html)]
+ _PyTorch Geometric_ implementation of the [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf) [[link](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv)]
+ Learning Local Search Heuristics for Boolean Satisfiability [[paper](https://papers.nips.cc/paper/9012-learning-local-search-heuristics-for-boolean-satisfiability.pdf), [code](https://github.com/emreyolcu/sat)]
+ PDP Framework for Neural Constraint Satisfaction Solving [[paper](https://arxiv.org/pdf/1903.01969.pdf), [code](https://github.com/microsoft/PDP-Solver)]

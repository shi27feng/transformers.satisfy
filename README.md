# _Propositional Satisfiability Problem_ (SAT) goes neural and deep
We'd like to use graph neural network to solve the SAT (more general _Constraint Satisfaction Problem_)

## General questions:
+ [How to convert a __CSP__ to _k_-__SAT__ ](https://cs.stackexchange.com/questions/23157/transformation-of-constraint-satisfaction-to-sat): For _SAT_, there are $n$ binary variables and $m$ constraints. Each constraint is associated with a $k$-tuple of (distinct) variables, for some $k$ depending on the constraint, along with a subset of {0,1} $k$ which is the allowed assignments for the $k$-tuple of variables.

For example, graph coloring, or rather, whether a given graph _G_ can be $x$-colored, can be viewed as a CSP. If there are $n$ vertices (we identify the vertex set with {1,…,n}) then we have n groups $x_1, \dots, x_n$ of $ \big \lceil \log_2 x \big \rceil $ variables. For each $i \in \{1,…,n\}$ there is a constraint on each group $x_i$ stating that the assignment for $x_i$ is the binary encoding of a number in the range {0, …, x − 1}. For any two connected vertices $(i,j)$ there is a constraint on both groups $x_i, x_j$ stating that $x_i \neq x_j$. This CSP is satisfiable iff _G_ is $x$-colorable.

In order to convert such a binary CSP to a SAT instance, we replace each constraint with the corresponding CNF. Continuing the example above, if $x=2$ then for any $i$,$j$ there is a constraint $x_i \neq x_j$ which we realize as $(x_i \vee \neg x_j) \wedge (\neg x_i \vee x_j)$.
The resulting CNF is satisfiable iff the CSP is. You can use the standard reduction to convert this CNF to a 3CNF if you wish.

We can also handle CSPs which are not binary, i.e., in which the variables are not binary but rather have some finite domain. The idea is similar to how I described coloring as a CSP above, and left to the reader.


## References:
+ Transformers are graph neural networks [[link](https://docs.dgl.ai/en/latest/tutorials/models/4_old_wines/7_transformer.html)]
+ _PyTorch Geometric_ implementation of the [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf) [[link](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv)]
+ Learning Local Search Heuristics for Boolean Satisfiability [[paper](https://papers.nips.cc/paper/9012-learning-local-search-heuristics-for-boolean-satisfiability.pdf), [code](https://github.com/emreyolcu/sat)]
+ PDP Framework for Neural Constraint Satisfaction Solving [[paper](https://arxiv.org/pdf/1903.01969.pdf), [code](https://github.com/microsoft/PDP-Solver)]

# Transformers Satisfy

## Data and DataLoader

Dataset are automatically downloaded from [SATLIB](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html)

    - Uniform Random 3-SAT (RND3SAT)
    - DIMACS Benchmark Instances (DIMACS)

### Example

```python
# How to use SATDataset
from data import SATDataset
ds = SATDataset('dataset', 'RND3SAT/uf50-218', use_negative=True)


Satisfied Cases ...
Downloading https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf50-218.tar.gz
Extracting dataset/raw/uf50-218.tar.gz
Unsatisfied Cases ...
Downloading https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uuf50-218.tar.gz
Extracting dataset/raw/uuf50-218.tar.gz
Processing...
processing uf50-0965.cnf: 100%|██████████| 2000/2000 [00:04<00:00, 487.37it/s]
Done!

```

# openmp-forest
Decision Tree and Random Forest implementation in C using OpenMP for parallelization. (INF01008 Assignment)

## Build

Compile the project using `make`:

```bash
make
```

## Run

Execute the compiled binary `bin/dt_program` passing the number of threads and the parallelization strategy:

```bash
./bin/dt_program [num_threads] [strategy]
```

### Strategies
- `s`: Sequential (no parallelization)
- `p`: Parallelize splits (node feature evaluations)
- `t`: Parallelize splits if the number of samples is large, otherwise parallelize tree nodes.
- `n`: Parallelize tree node creation only.
- `P`: Parallelize both splits and node creation.

### Usage Example
Run with 4 threads using the split parallelization strategy (`p`):
```bash
./bin/dt_program 4 p
```

Alternatively, via `make run`:
```bash
make run THREADS=4 STRATEGY=p
```

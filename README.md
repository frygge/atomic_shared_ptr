# atomic_shared_ptr

This repository provides a header-only implementation for lock-free atomic shared pointers.
The description of the algorithm including a proof of its correctness and a throughput evaluation can be found in the [paper](atomic_sptr.pdf).

## Evaluation

The evaluation provided in the paper was done using the measure tool in the test/ directory.

Example: The following commands build the binary for measuring in a subdirectory.
The measurement is the run for 1 to 32 workers and 1 to 56 variables but not with `no_contention` (i.e. only in contention mode).

```
mkdir build && cd build
cmake ..
make
./measure -workers 1 +workers 32 -vars 1 +vars 64 -no_contention | tee output.txt
```

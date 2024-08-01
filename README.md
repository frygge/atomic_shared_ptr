# atomic_shared_ptr

This repository provides a header-only implementation for lock-free atomic shared pointers.
The description of the algorithm including a proof of its correctness and a throughput evaluation can be found in the [paper](atomic_sptr.pdf). The paper has been reviewed and published here:
Schäfer, J.P. (2024). Faster Lock-Free Atomic Shared Pointers. In: Intelligent Computing. SAI 2024. Lecture Notes in Networks and Systems, vol 1017. Springer, Cham. https://doi.org/10.1007/978-3-031-62277-9_2

## Evaluation

The evaluation provided in the paper was done using the measure tool in the `test/` directory.

Example: The following commands build the binary for measuring in a subdirectory.
The measurement is the run for 2 to 4 workers and 1 to 3 variables but not with `no_contention` (i.e. only in contention mode).

```bash
mkdir build && cd build
cmake ..
make
./measure -workers 2 +workers 4 -vars 1 +vars 3 -no_contention | tee ../output.txt
```

This command will run for ~4,5 minutes, no matter what machine is used.
See the paper for details.

To post-process the `output.txt`, use the `post-process_measurement.sh` script in the `test/` directory.

```bash
mkdir my_results && cd my_results
../test/post-process_measurement.sh ../output.txt my_machine_prefix
```

This will create one file for each single experiment, i.e., for each combination of library and operation.
The first column contains a '-'-serparated tuple of #threads and #vars; thus, joining multiple files on the first column and then replacing dash ('-') by space (' ') yields a gnuplot friendly output.

## Citation

If you use this work, please cite:

```bibtex
@InProceedings{10.1007/978-3-031-62277-9_2,
  author="Sch{\"a}fer, J{\"o}rg P.",
  title="Faster Lock-Free Atomic Shared Pointers",
  booktitle="Intelligent Computing",
  year="2024",
  publisher="Springer Nature Switzerland",
  pages="18--38",
  isbn="978-3-031-62277-9"
}
```

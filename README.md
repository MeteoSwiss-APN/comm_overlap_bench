# Communication Overlap Benchmark

This sofware is intended to run experiments that overlap communication and computation in order collect performance data. 
It runs a toy model where GPU kernels are overlap with MPI GPU communication. 
It provides a synchornous mode (where the host timeline is block until the communication has finished) and an aysnchronous 
mode where communication is split into Start and Wait methods wrapping the calls to the kernel launch. 

The following picture shows an example of flow of the code for a simple communication pattern:
![Example toy model testing overlap of comm/comp](doc/code1.png?raw=true "Optional Title")

## Command Line Arguments

* `--ie NN` domain size in x. Default: 128
* `--je NN` domain size in y. Default: 128
* `--ke NN` domain size in z. Default: 60
* `--sync` enable synchronous communication.
* `--nocomm` disable communication.
* `--nocomp` disable computation.
* `--nh NN` the number of halo updates. Default: 2
* `-n NN` the number of benchmark repetetitions. Default: 5000
* `--inorder` enable in order halo exchanges. 

# Disclaimer

Code needs cleanup...

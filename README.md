# Communication Overlap Benchmark

This sofware is intended to run experiments that overlap communication and computation in order collect performance data. 
It runs a toy model where GPU kernels are overlap with MPI GPU communication. 
It provides a synchornous mode (where the host timeline is block until the communication has finished) and an aysnchronous 
mode where communication is split into Start and Wait methods wrapping the calls to the kernel launch. 

The following picture shows an example of flow of the code for a simple communication pattern:
![Example toy model testing overlap of comm/comp](doc/code1.png?raw=true "Optional Title")

## Requirements

- CMake
- CUDA
- MPI
- Boost: Optional

## Building

Building is straightforward. Create a build directory and run CMake:

    mkdir build
    cd build
    cmake ..

The build contains multiple options for building. 

- `-DENABLE_TIMER=ON` will enable the boost timers to measure the performance. This requires boost. 
- `-DENABLE_MPI_TIMER=ON` will enable MPI timers.
- `-DVERBOSE=1` will print extra information when running the code.
- `-DMPI_VENDOR` allows setting an MPI vendor. Typical values are: `unknown` `mvapich2` or `openmpi`. Enabling this variables will simply increase the debug output.

A typical build can be achieved with:

    mkdir build
    cd build
    cmake .. -DVERBOSE=1 -DENABLE_TIMER=1


## Running
 
Simply call 

    src/comm_overlap_benchmark 

to run the code. The code can be run in several configurations:

    comm_overlap_benchmark [--ie isize] [--je jsize] [--ke ksize] \
                           [--sync] [--nocomm] [--nocomp] \
                           [--nh nhaloupdates] [-n nrepetitions] [--inorder]

### Command Line Arguments

* `--ie NN` domain size in x. Default: 128
* `--je NN` domain size in y. Default: 128
* `--ke NN` domain size in z. Default: 60
* `--sync` enable synchronous communication.
* `--nocomm` disable communication.
* `--nocomp` disable computation.
* `--nh NN` the number of halo updates. Default: 2
* `-n NN` the number of benchmark repetetitions. Default: 5000
* `--inorder` enable in order halo exchanges. 

### Typical configurations

Running without any MPI configuration (e.g. test the perofrmance of the CUDA kernel):
    
    src/comm_overlap_benchmark --nocomm

Profiling the kernel with CUDA:

     nvprof --analysis-metrics -o benchmark.prof ./src/comm_overlap_benchmark --nocomm --nrep=1


# Disclaimer

Code needs cleanup...

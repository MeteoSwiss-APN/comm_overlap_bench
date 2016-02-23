# comm_overlap_bench

This sofware is intended to run experiments that overlap communication and computation in order collect performance data. 
It runs a toy model where GPU kernels are overlap with MPI GPU communication. 
It provides a synchornous mode (where the host timeline is block until the communication has finished) and an aysnchronous 
mode where communication is split into Start and Wait methods wrapping the calls to the kernel launch. 

The following picture shows an example of flow of the code for a simple communication pattern:
![Example toy model testing overlap of comm/comp](doc/code1.png?raw=true "Optional Title")

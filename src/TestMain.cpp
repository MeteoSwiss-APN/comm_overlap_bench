#include <algorithm>
#include "gtest/gtest.h"
#include "Options.h"
#include "UnittestEnvironment.h"

// override main in order to read further program arguments
int main(int argc, char **argv) 
{
    const char* env_p = std::getenv("SLURM_PROCID");
    if(!env_p) {
        std::cout << "SLURM_PROCID not set" << std::endl;
        exit (EXIT_FAILURE);
    }

    int numGPU;
    cudaError_t error = cudaGetDeviceCount(&numGPU);
    if(error)  {
        std::cout << "CUDA ERROR " << std::endl;
        exit(EXIT_FAILURE);
    }

    error = cudaSetDevice(atoi(env_p)%numGPU);
    if(error)  {
        std::cout << "CUDA ERROR " << std::endl;
        exit(EXIT_FAILURE);
    }

}

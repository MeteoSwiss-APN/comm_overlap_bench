#include <stdexcept>
#include <iostream>
#include <cmath>
#include <boost/lexical_cast.hpp>
#include <boost/timer/timer.hpp>
#include "HorizontalDiffusionSA.h"
#include "Definitions.h"
#include "Options.h"
#include <mpi.h>
#include "MPIHelper.h"

#include "IJKSize.h"

#ifdef __CUDA_BACKEND__
#include "cuda_profiler_api.h"
#endif

void readOptions(int argc, char** argv)
{
    Options::setCommandLineParameters(argc, argv);

    const int& rank = Options::get<int>("rank");

    if (rank == 0) {
        std::cout << "StandaloneStencilsCUDA\n\n";
        std::cout << "usage: StandaloneStencilsCUDA [--ie isize] [--je jsize] [--ke ksize] \\\n"
                  << "                              [--sync] [--nocomm] [--nocomp] \\\n"
                  << "                              [--nh nhaloupdates] [-n nrepetitions] [--inorder]\n"
                  << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    Options::parse("isize",  "--ie",       128);
    Options::parse("jsize",  "--je",       128);
    Options::parse("ksize",  "--ke",        60);

    Options::parse("sync",   "--sync",   false);
    Options::parse("nocomm", "--nocomm", false);
    Options::parse("nocomp", "--nocomp", false);

    Options::parse("nhaloupdates", "--nh",                      2);
    Options::parse("nrep",         "-n", cNumBenchmarkRepetitions);

    Options::parse("inorder", "--inorder", false);
}

void setupDevice()
{
    const int& rank = Options::get<int>("rank");

#ifdef MVAPICH2
    const char* env_p = std::getenv("SLURM_PROCID");
    if(!env_p) {
        std::cout << "SLURM_PROCID not set" << std::endl;
        exit (EXIT_FAILURE);
    }

    const char* local_rank = std::getenv("MV2_COMM_WORLD_LOCAL_RANK");
    if (local_rank) {
        MPIHelper::print("MV2_COMM_WORLD_LOCAL_RANK: ", std::string(local_rank)+", ", 9999);
        env_p = local_rank;
    }
#elif OPENMPI
    const char* env_p = std::getenv("OMPI_COMM_WORLD_RANK");
    if(!env_p) {
        std::cout << "OMPI_COMM_WORLD_RANK not set" << std::endl;
        exit (EXIT_FAILURE);
    }
    MPIHelper::print("OMPI_COMM_WORLD_RANK: ", std::string(env_p)+", ", 9999);

#else
    const char* env_p = "0";
#endif

    const char* visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
    if (visible_devices) {
        MPIHelper::print("CUDA_VISIBLE_DEVICES: ", "["+std::to_string(rank)+": "+std::string(visible_devices)+"] ", 9999);
    }
    int numGPU;
    cudaError_t error = cudaGetDeviceCount(&numGPU);
    if(error)  {
        std::cout << "Rank: "+std::to_string(rank)+"CUDA ERROR: No device found " << std::endl;
        exit(EXIT_FAILURE);
    }

    error = cudaSetDevice(atoi(env_p)%numGPU);
    if(error)  {
        std::cout << "Rank: "+std::to_string(rank)+"CUDA ERROR: Could not set device " << std::to_string(atoi(env_p)%numGPU)
                  << std::endl;
        exit(EXIT_FAILURE);
    }
    
    int device;
    error = cudaGetDevice(&device);
    if(error)  {
        std::cout << "Rank: "+std::to_string(rank)+"CUDA ERROR: Could not get configured device." << std::endl;
        exit(EXIT_FAILURE);
    }
    MPIHelper::print("Configured CUDA Devices: ", "["+std::to_string(rank)+": "+std::to_string(device)+"] ", 9999);
}

void printSlurmInfo() {

    const int& rank = Options::get<int>("rank");

    const char* jobid = std::getenv("SLURM_JOBID");
    if (jobid != 0 && rank == 0) {
        std::cout << "SLURM_JOBID: " << jobid << std::endl;
    }
    const char* procid = std::getenv("SLURM_PROCID");
    if (procid != 0) {
        MPIHelper::print("SLURM_PROCID: ", std::string(procid)+", ", 9999);
    }
    const char* cpu_bind = std::getenv("SLURM_CPU_BIND");
    if (cpu_bind != 0) {
        MPIHelper::print("SLURM_CPU_BIND: ", "["+std::to_string(rank)+": "+std::string(cpu_bind)+"] ", 9999);
    }
    const char* nodename = std::getenv("SLURMD_NODENAME");
    if (nodename != 0) {
        MPIHelper::print("SLURMD_NODENAME: ", std::string(nodename)+", ", 9999);
    }
}

void init_mpi(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Options::set("rank", rank);
}

int main(int argc, char** argv)
{
    init_mpi(argc, argv);
    readOptions(argc, argv);
    setupDevice();
    printSlurmInfo();

    IJKSize domain(Options::get<int>("isize"),
                   Options::get<int>("jsize"),
                   Options::get<int>("ksize"));
    auto repository = std::shared_ptr<Repository>(new Repository(domain));

    const int& rank = Options::get<int>("rank");
    if (Options::get<int>("rank") == 0) {
        std::cout << "\nCONFIGURATION " << std::endl;
        std::cout << "====================================" << std::endl;
        std::cout << "Domain : [" << domain.isize << "," << domain.jsize << "," << domain.ksize << "]" << std::endl;
        std::cout << "Sync? : " << Options::get<bool>("sync") << std::endl;
        std::cout << "NoComm? : " << Options::get<bool>("nocomm") << std::endl;
        std::cout << "NoComp? : " << Options::get<bool>("nocomp") << std::endl;
        std::cout << "Number Halo Exchanges : " << Options::get<int>("nhaloupdates") << std::endl;
        std::cout << "Number benchmark repetitions : " << Options::get<int>("nrep") << std::endl;
        std::cout << "In Order halo exchanges? : " << Options::get<bool>("inorder") << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int deviceId;
    if( cudaGetDevice(&deviceId) != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }

#ifdef MVAPICH2
    if (rank == 0)
        std::cout << "Compiled for mvapich2" << std::endl;
#elif OPENMPI
    if (rank == 0)
        std::cout << "Compiled for openmpi" << std::endl;
#else
    // Default proc
    const char* env_p = "0";
#endif

    boost::timer::cpu_timer cpu_timer;

    // Generate a horizontal diffusion operator
    HorizontalDiffusionSA horizontalDiffusionSA(repository);
    cudaDeviceSynchronize();
#ifdef __CUDA_BACKEND__
    cudaProfilerStart();
#endif

    bool sync = Options::get<bool>("sync");
    bool nocomm = Options::get<bool>("nocomm");
    bool inOrder = Options::get<bool>("inorder");
    bool nocomp = Options::get<bool>("nocomp");
    int nHaloUpdates = Options::get<int>("nhaloupdates");
    int nRep = Options::get<int>("nrep");

    MPI_Barrier(MPI_COMM_WORLD);

    cpu_timer.start();

    // Benchmark!
    for(int i=0; i < nRep; ++i) {
        int numGPU;
        cudaGetDeviceCount(&numGPU);
        // flush cache between calls to horizontal diffusion stencil
        if(i!=0 && !sync && !nocomm && !inOrder) {
            for(int c=0; c < nHaloUpdates ; ++c) {
                horizontalDiffusionSA.WaitHalos(c);
            }
        }
        if(!nocomp) {
            horizontalDiffusionSA.Apply();
        }

        if(inOrder) {
            for(int c=0; c < nHaloUpdates; ++c) {
                if(!nocomm){
                    if(!sync)
                        horizontalDiffusionSA.StartHalos(c);
                    else
                        horizontalDiffusionSA.ApplyHalos(c);
                }
                repository->swap();
                if(!nocomp)
                    horizontalDiffusionSA.Apply();

                if(!nocomm){
                    if(!sync)
                        horizontalDiffusionSA.WaitHalos(c);
                }
            }

        }
        else {
            for(int c=0; c < nHaloUpdates ; ++c) {
                if(!nocomm){
                    if(!sync)
                        horizontalDiffusionSA.StartHalos(c);
                    else
                        horizontalDiffusionSA.ApplyHalos(c);
                }
                repository->swap();
                if(!nocomp)
                    horizontalDiffusionSA.Apply();
            }
        }
    }
    if(!nocomm && !sync && !inOrder){
        for(int c=0; c < nHaloUpdates ; ++c) {
            horizontalDiffusionSA.WaitHalos(c);
        }
    }
    cudaDeviceSynchronize();
#ifdef __CUDA_BACKEND__
    cudaProfilerStop();
#endif

    cpu_timer.stop();
    boost::timer::cpu_times elapsed = cpu_timer.elapsed();

    double total_time = ((double) elapsed.wall)/1000000000.0;
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    int rank_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    std::vector<double> total_time_g(num_ranks);

    MPI_Gather(&total_time, 1, MPI_DOUBLE, &(total_time_g[0]), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank_id==0) {
        double avg = 0.0;
        double rms = 0.0;
        for(int i=0; i < num_ranks; ++i)
        {
            avg += total_time_g[i];
        }
        avg /= (double)num_ranks;
        for(int i=0; i < num_ranks; ++i)
        {
            rms += (total_time_g[i]-avg)*(total_time_g[i]-avg);
        }

        rms /= (double)num_ranks;
        rms = std::sqrt(rms);

        std::cout <<"ELAPSED TIME : " << avg << " +- + " << rms << std::endl;
    }
    MPI_Finalize();
}

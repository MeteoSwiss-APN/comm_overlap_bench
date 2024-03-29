#include "Definitions.h"
#include "HorizontalDiffusionSA.h"
#include "MPIHelper.h"
#include "Options.h"
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <sched.h>
#include <nvml.h>

#include "IJKSize.h"

#ifdef ENABLE_BOOST_TIMER
#include <boost/timer/timer.hpp>
#endif

#ifdef CUDA_BACKEND
#include <cuda_profiler_api.h>
#endif

void readOptions(int argc, char** argv) {
    Options::setCommandLineParameters(argc, argv);

    const int& rank = Options::getInt("rank");

#ifdef VERBOSE
    if (rank == 0) {
        std::cout << "StandaloneStencilsCUDA\n\n";
        std::cout << "usage: StandaloneStencilsCUDA [--ie isize] [--je jsize] [--ke ksize] \\\n"
                  << "                              [--sync] [--nocomm] [--nocomp] \\\n"
                  << "                              [--nh nhaloupdates] [-n nrepetitions] [--inorder]\n"
                  << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    Options::parse("isize", "--ie", 128);
    Options::parse("jsize", "--je", 128);
    Options::parse("ksize", "--ke", 60);

    Options::parse("sync", "--sync", false);
    Options::parse("nocomm", "--nocomm", false);
    Options::parse("nocomp", "--nocomp", false);

    Options::parse("nhaloupdates", "--nh", 2);
    Options::parse("nrep", "-n", cNumBenchmarkRepetitions);

    Options::parse("inorder", "--inorder", false);
}

void setupDevice() {
    const int& rank = Options::getInt("rank");

#ifdef CUDA_BACKEND
    const char* visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
    if (visible_devices) {
        MPIHelper::print(
            "CUDA_VISIBLE_DEVICES: ", "[" + std::to_string(rank) + ": " + std::string(visible_devices) + "] ", 9999);
    }
    int numGPU;
    cudaError_t error = cudaGetDeviceCount(&numGPU);
    if (error) {
        std::cout << "Rank: " + std::to_string(rank) + "CUDA ERROR: No device found " << std::endl;
        exit(EXIT_FAILURE);
    }

    nvmlInit_v2();

    char uuid[NVML_DEVICE_UUID_BUFFER_SIZE];
    int cudaDevice = std::stoi(std::getenv("CUDA_VISIBLE_DEVICES"));

    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex_v2(cudaDevice, &device);
    nvmlDeviceGetUUID (device, uuid, sizeof(uuid) );  


    nvmlShutdown();

    MPIHelper::print(
        "Configured CUDA Devices: ", "{'rank': " + std::to_string(rank) + ", 'device': " + std::to_string(cudaDevice) + 
        ", 'GPU_uuid': "+ std::string(uuid) +", 'CPU_id:'"+std::to_string(sched_getcpu()), 9999);
#else
    if (rank == 0) {
        std::cout << "CUDA Mode Disabled" << std::endl;
    }
#endif
}

void printSlurmInfo() {

    const int& rank = Options::getInt("rank");

    const char* jobid = std::getenv("SLURM_JOBID");
    if (jobid != 0 && rank == 0) {
        std::cout << "SLURM_JOBID: " << jobid << std::endl;
    }
    const char* procid = std::getenv("SLURM_PROCID");
    if (procid != 0) {
        MPIHelper::print("SLURM_PROCID: ", std::string(procid) + ", ", 9999);
    }
    const char* cpu_bind = std::getenv("SLURM_CPU_BIND");
    if (cpu_bind != 0) {
        MPIHelper::print("SLURM_CPU_BIND: ", "[" + std::to_string(rank) + ": " + std::string(cpu_bind) + "] ", 9999);
    }
    const char* nodename = std::getenv("SLURMD_NODENAME");
    if (nodename != 0) {
        MPIHelper::print("SLURMD_NODENAME: ", std::string(nodename) + ", ", 9999);
    }
    const char* usecuda1 = std::getenv("MV2_USE_CUDA");
    if (usecuda1 != 0 && rank == 0) {
        std::cout << "MV2_USE_CUDA: " << usecuda1 << std::endl;
    }
    const char* usecuda2 = std::getenv("MPICH_RDMA_ENABLED_CUDA");
    if (usecuda2 != 0 && rank == 0) {
        std::cout << "MPICH_RDMA_ENABLED_CUDA: " << usecuda2 << std::endl;
    }
}

void init_mpi(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Options::set("rank", rank);
}

int main(int argc, char** argv) {
    init_mpi(argc, argv);
    readOptions(argc, argv);
    setupDevice();
    printSlurmInfo();

    IJKSize domain(Options::getInt("isize"), Options::getInt("jsize"), Options::getInt("ksize"));
    auto repository = std::shared_ptr< Repository >(new Repository(domain));

    const int& rank = Options::getInt("rank");
    if (rank == 0) {
        std::cout << "\nCONFIGURATION " << std::endl;
        std::cout << "====================================" << std::endl;
        std::cout << "Domain : [" << domain.isize << "," << domain.jsize << "," << domain.ksize << "]" << std::endl;
        std::cout << "Sync? : " << Options::getBool("sync") << std::endl;
        std::cout << "NoComm? : " << Options::getBool("nocomm") << std::endl;
        std::cout << "NoComp? : " << Options::getBool("nocomp") << std::endl;
        std::cout << "Number Halo Exchanges : " << Options::getInt("nhaloupdates") << std::endl;
        std::cout << "Number benchmark repetitions : " << Options::getInt("nrep") << std::endl;
        std::cout << "In Order halo exchanges? : " << Options::getBool("inorder") << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

#ifdef CUDA_BACKEND
    int deviceId;
    if (cudaGetDevice(&deviceId) != cudaSuccess) {
        std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }
#endif

#ifdef VERBOSE
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
#endif

    // Generate a horizontal diffusion operator
    HorizontalDiffusionSA horizontalDiffusionSA(repository);

#ifdef CUDA_BACKEND
    cudaDeviceSynchronize();
    cudaProfilerStart();
#endif

    bool sync = Options::getBool("sync");
    bool nocomm = Options::getBool("nocomm");
    bool inOrder = Options::getBool("inorder");
    bool nocomp = Options::getBool("nocomp");
    int nHaloUpdates = Options::getInt("nhaloupdates");
    int nRep = Options::getInt("nrep");

    MPI_Barrier(MPI_COMM_WORLD);

#ifdef ENABLE_BOOST_TIMER
    boost::timer::cpu_timer cpu_timer;
    cpu_timer.start();
#elif ENABLE_MPI_TIMER
    double mpi_starttime = MPI_Wtime();
#endif

    // Benchmark!
    for (int i = 0; i < nRep; ++i) {
        // flush cache between calls to horizontal diffusion stencil
        if (i != 0 && !sync && !nocomm && !inOrder) {
            for (int c = 0; c < nHaloUpdates; ++c) {
                horizontalDiffusionSA.WaitHalos(c);
            }
        }
        if (!nocomp) {
            horizontalDiffusionSA.Apply();
        }

        if (inOrder) {
            for (int c = 0; c < nHaloUpdates; ++c) {
                if (!nocomm) {
                    if (!sync)
                        horizontalDiffusionSA.StartHalos(c);
                    else
                        horizontalDiffusionSA.ApplyHalos(c);
                }
                repository->swap();
                if (!nocomp)
                    horizontalDiffusionSA.Apply();

                if (!nocomm) {
                    if (!sync)
                        horizontalDiffusionSA.WaitHalos(c);
                }
            }

        } else {
            for (int c = 0; c < nHaloUpdates; ++c) {
                if (!nocomm) {
                    if (!sync)
                        horizontalDiffusionSA.StartHalos(c);
                    else
                        horizontalDiffusionSA.ApplyHalos(c);
                }
                repository->swap();
                if (!nocomp)
                    horizontalDiffusionSA.Apply();
            }
        }
    }
    if (!nocomm && !sync && !inOrder) {
        for (int c = 0; c < nHaloUpdates; ++c) {
            horizontalDiffusionSA.WaitHalos(c);
        }
    }

#ifdef CUDA_BACKEND
    cudaProfilerStop();
    cudaDeviceSynchronize();
#endif

#ifdef ENABLE_BOOST_TIMER
    cpu_timer.stop();
    boost::timer::cpu_times elapsed = cpu_timer.elapsed();
    double total_time = ((double)elapsed.wall) / 1000000000.0;
#elif ENABLE_MPI_TIMER
    double mpi_endtime = MPI_Wtime();
    double total_time = mpi_endtime - mpi_starttime;
#endif

#if defined(ENABLE_BOOST_TIMER) || defined(ENABLE_MPI_TIMER)
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    int rank_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    std::vector< double > total_time_g(num_ranks);

    MPI_Gather(&total_time, 1, MPI_DOUBLE, &(total_time_g[0]), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank_id == 0) {
        double avg = 0.0;
        double rms = 0.0;
        for (int i = 0; i < num_ranks; ++i) {
            avg += total_time_g[i];
        }
        avg /= (double)num_ranks;
        for (int i = 0; i < num_ranks; ++i) {
            rms += (total_time_g[i] - avg) * (total_time_g[i] - avg);
        }

        rms /= (double)num_ranks;
        rms = std::sqrt(rms);

        std::cout << "ELAPSED TIME: " << avg << " +- + " << rms << std::endl;
    }
#else
    std::cout << "Timers disabled: Enable by compiling with -DENABLE_BOOST_TIMER (Boost timers) or -DENABLE_MPI_TIMER "
                 "(MPI_Wtime)"
              << std::endl;
#endif

    MPI_Finalize();
}

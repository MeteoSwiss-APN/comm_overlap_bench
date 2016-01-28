#include <stdexcept>
#include <sstream>
#include <cmath>
#include <boost/lexical_cast.hpp>
#include <boost/timer/timer.hpp>
#include "gtest/gtest.h"
#include "CacheFlush.h"
#include "HorizontalDiffusionSA.h"
#include "HoriDiffReference.h"
#include "Definitions.h"
#include "UnittestEnvironment.h"
#include "HaloUpdateManager.h"
#include "Options.h"
#include "mpi.h"

#ifdef __CUDA_BACKEND__
#include "cuda_profiler_api.h"
#endif

// method parsing a string option
int parseIntOption(int argc, char **argv, std::string option, int defaultValue)
{
    int result = defaultValue;
    for(int i = 0; i < argc; ++i)
    {
        if(argv[i] == option && i+1 < argc)
        {
            result = boost::lexical_cast<int>(argv[i+1]);
            break;
        }
    }
    return result;
}

// method parsing a string option
std::string parseStringOption(int argc, char **argv, std::string option, std::string defaultValue)
{
    std::string result = defaultValue;
    for(int i = 0; i < argc; ++i)
    {
        if(argv[i] == option && i+1 < argc)
        {
            result = argv[i+1];
            break;
        }
    }
    return result;
}

// method parsing a boolean option
bool parseBoolOption(int argc, char **argv, std::string option)
{
    bool result = false;
    for(int i = 0; i < argc; ++i)
    {
        if(argv[i] == option)
        {
            result = true;
            break;
        }
    }
    return result;
}

void readOptions(int argc, char** argv)
{
    std::cout << "HP2CDycoreUnittest\n\n";
    std::cout << "usage: HP2CDycoreUnittest -p <dataPath>" << "\n";
    for(int i=0; i < argc; ++i)
        std::cout << std::string(argv[i]) << std::endl;

    // parse additional command options
    int iSize = parseIntOption(argc, argv, "--ie", 128);
    int jSize = parseIntOption(argc, argv, "--je", 128);
    int kSize = parseIntOption(argc, argv, "--ke", 60);
    bool sync = parseBoolOption(argc, argv, "--sync");
    bool nocomm = parseBoolOption(argc, argv, "--nocomm");
    bool nocomp = parseBoolOption(argc, argv, "--nocomp");
    bool nostella = parseBoolOption(argc, argv, "--nostella");
    bool nogcl = parseBoolOption(argc, argv, "--nogcl");
    int nHaloUpdates = parseIntOption(argc, argv, "--nh", 2);
    int nRep = parseIntOption(argc, argv, "-n", cNumBenchmarkRepetitions);
    bool inOrder = parseBoolOption(argc, argv, "--inorder");

    IJKSize domain;
    domain.Init(iSize, jSize, kSize);
    Options::getInstance().domain_ = domain;
    Options::getInstance().sync_ = sync;
    Options::getInstance().nocomm_ = nocomm;
    Options::getInstance().nocomp_ = nocomp;
    Options::getInstance().nostella_ = nostella;
    Options::getInstance().nogcl_ = nogcl;
    Options::getInstance().nHaloUpdates_ = nHaloUpdates;
    Options::getInstance().nRep_ = nRep;
    Options::getInstance().inOrder_ = inOrder;

}

void setupDevice()
{
#ifdef MVAPICH2
    const char* env_p = std::getenv("SLURM_PROCID");
    if(!env_p) {
        std::cout << "SLURM_PROCID not set" << std::endl;
        exit (EXIT_FAILURE);
    }
#else
    const char* env_p = std::getenv("OMPI_COMM_WORLD_RANK");
    if(!env_p) {
        std::cout << "OMPI_COMM_WORLD_RANK not set" << std::endl;
        exit (EXIT_FAILURE);
    }
#endif

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

int main(int argc, char** argv)
{

    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    readOptions(argc, argv);
    setupDevice();


    HoriDiffRepository* pRepository_ = &UnittestEnvironment::getInstance().repository();

    IJKSize domain_ = UnittestEnvironment::getInstance().calculationDomain();

    HoriDiffReference ref_;

    ref_.Init(*pRepository_);
    ref_.Generate();
    const ErrorMetric* pMetric_ = &UnittestEnvironment::getInstance().metric();

    std::cout << "CONFIGURATION " << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Domain : [" << Options::getInstance().domain_.iSize() << "," << Options::getInstance().domain_.jSize() << "," <<Options::getInstance().domain_.kSize() << "]" << std::endl;
    std::cout << "Sync? : " << Options::getInstance().sync_ << std::endl;
    std::cout << "NoComm? : " << Options::getInstance().nocomm_ << std::endl;
    std::cout << "NoComp? : " << Options::getInstance().nocomp_ << std::endl;
    std::cout << "NoStella? : " << Options::getInstance().nostella_ << std::endl;
    std::cout << "NoGCL? : " << Options::getInstance().nogcl_ << std::endl;
    std::cout << "Number Halo Exchanges : " << Options::getInstance().nHaloUpdates_ << std::endl;
    std::cout << "Number benchmark repetitions : " << Options::getInstance().nRep_ << std::endl;
    std::cout << "In Order halo exchanges? : " << Options::getInstance().inOrder_ << std::endl;

    int deviceId;
    if( cudaGetDevice(&deviceId) != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }
    std::cout << "Device ID :" << deviceId << std::endl;
    const char* env_p = std::getenv("SLURM_PROCID");
    std::cout << "SLURM_PROCID :" << env_p << std::endl;
    env_p = std::getenv("OMPI_COMM_WORLD_RANK");
    std::cout << "OMPI_COMM_WORLD_RANK :" << env_p<< std::endl;

#ifdef MVAPICH2
    std::cout << "Compiled for mvapich2" << std::endl;
#else
    std::cout << "Compiled for openmpi" << std::endl;
#endif

    boost::timer::cpu_timer cpu_timer;
    // set up cache flusher for x86
    CacheFlush cacheFlusher(cCacheFlusherSize);

    // generate a reference field that contain the output of a horizontal diffusion operator
    HorizontalDiffusionSA horizontalDiffusionSA;
    horizontalDiffusionSA.Init(*pRepository_, UnittestEnvironment::getInstance().communicationConfiguration());

    IJBoundary applyB;
    applyB.Init(-1,1,-1,1);
    horizontalDiffusionSA.Apply(applyB);
//    horizontalDiffusionSA.ApplyHalos();
    pRepository_->Swap();
    horizontalDiffusionSA.Apply();

    IJKRealField& outField = pRepository_->u_out(0);
    IJKRealField& refField = pRepository_->refField();

    // we only verify the stencil once (before starting the real benchmark)
    IJKBoundary checkBoundary;
    checkBoundary.Init(0, 0, 0, 0, 0, 0);

    horizontalDiffusionSA.Apply();
    cudaDeviceSynchronize();
    horizontalDiffusionSA.ResetMeters();
#ifdef __CUDA_BACKEND__
    cudaProfilerStart();
#endif

    cpu_timer.start();

    for(int i=0; i < Options::getInstance().nRep_; ++i) {
        int numGPU;
        cudaGetDeviceCount(&numGPU);
        // flush cache between calls to horizontal diffusion stencil
        if(i!=0 && !Options::getInstance().sync_ && !Options::getInstance().nocomm_ && !Options::getInstance().inOrder_) {
            for(int c=0; c < Options::getInstance().nHaloUpdates_ ; ++c) {
                horizontalDiffusionSA.WaitHalos(c);
            }
        }
        if(!Options::getInstance().nocomp_) {
            horizontalDiffusionSA.Apply();
        }

        if(Options::getInstance().inOrder_) {
            for(int c=0; c < Options::getInstance().nHaloUpdates_ ; ++c) {
                if(!Options::getInstance().nocomm_){
                    if(!Options::getInstance().sync_)
                        horizontalDiffusionSA.StartHalos(c);
                    else
                        horizontalDiffusionSA.ApplyHalos(c);
                }
                pRepository_->Swap();
                
                if(!Options::getInstance().nocomp_)
                    horizontalDiffusionSA.Apply();

                if(!Options::getInstance().nocomm_){
                    if(!Options::getInstance().sync_)
                        horizontalDiffusionSA.WaitHalos(c);
                }
            }

        }
        else {
            for(int c=0; c < Options::getInstance().nHaloUpdates_ ; ++c) {
                if(!Options::getInstance().nocomm_){
                    if(!Options::getInstance().sync_)
                        horizontalDiffusionSA.StartHalos(c);
                    else
                        horizontalDiffusionSA.ApplyHalos(c);
                }
                pRepository_->Swap();
                if(!Options::getInstance().nocomp_)
                    horizontalDiffusionSA.Apply();
            }
        }
    }
    if(!Options::getInstance().nocomm_ && !Options::getInstance().sync_ && !Options::getInstance().inOrder_){
        for(int c=0; c < Options::getInstance().nHaloUpdates_ ; ++c) {
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
        total_time_g[3] = 5;
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

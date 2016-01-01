#include <stdexcept>
#include <sstream>
#include <cmath>
#include <boost/timer/timer.hpp>
#include "gtest/gtest.h"
#include "CacheFlush.h"
#include "HorizontalDiffusionSA.h"
#include "HoriDiffReference.h"
#include "Definitions.h"
#include "UnittestEnvironment.h"
#include "HaloUpdateManager.h"
#include "Options.h"

#ifdef __CUDA_BACKEND__
#include "cuda_profiler_api.h"
#endif

class HoriDiffBenchmark : public ::testing::Test
{
protected:

    // references for quick access
    HoriDiffRepository* pRepository_;
    IJKSize domain_;
    HoriDiffReference ref_;
    const ErrorMetric* pMetric_;

    virtual void SetUp()
    {
        pRepository_ = &UnittestEnvironment::getInstance().repository();
        domain_ = UnittestEnvironment::getInstance().calculationDomain();
        ref_.Init(*pRepository_);
        ref_.Generate();
        pMetric_ = &UnittestEnvironment::getInstance().metric();
    }
};

TEST_F(HoriDiffBenchmark, SingleVar)
{

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
    FieldCollection collection;
    collection.AddOutputReference("field", outField, refField, *pMetric_, checkBoundary);
    ASSERT_TRUE(collection.Verify());

    horizontalDiffusionSA.Apply();
    horizontalDiffusionSA.ResetMeters();
#ifdef __CUDA_BACKEND__
    cudaProfilerStart();
#endif

    cpu_timer.start();

    for(int i=0; i < cNumBenchmarkRepetitions; ++i) {
        // flush cache between calls to horizontal diffusion stencil
        if(i!=0 && !Options::getInstance().sync_ && !Options::getInstance().nocomm_)
            horizontalDiffusionSA.WaitHalos();
        horizontalDiffusionSA.Apply();
        if(!Options::getInstance().nocomm_){
            if(!Options::getInstance().sync_)
                horizontalDiffusionSA.StartHalos();
            else
                horizontalDiffusionSA.ApplyHalos();
        }
        pRepository_->Swap();
        horizontalDiffusionSA.Apply();
    }
    if(!Options::getInstance().nocomm_){
        horizontalDiffusionSA.WaitHalos();
    }

    cpu_timer.stop();
    boost::timer::cpu_times elapsed = cpu_timer.elapsed();

    double total_time = ((double) elapsed.wall)/1000000000.0;
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    int rank_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    std::vector<double> total_time_g(num_ranks);

    MPI_Gather(&total_time, 1, MPI_DOUBLE, &total_time_g[0], num_ranks, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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


std::cout << "NUMRAN " << num_ranks<<std::endl;
        rms /= (double)num_ranks;
        rms = std::sqrt(rms);

        std::cout <<"ELAPSED TIME : " << avg << " +- + " << rms << std::endl;
    }
//    std::cout <<"ELAPSED TIME : " << total_time << std::endl;

#ifdef __CUDA_BACKEND__
    cudaProfilerStop();
#endif

//    std::cout << horizontalDiffusionSA.stencil().performanceMeter().ToString() << std::endl;
}



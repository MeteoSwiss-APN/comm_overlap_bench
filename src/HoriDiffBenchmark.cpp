#include <stdexcept>
#include <sstream>
#include "gtest/gtest.h"
#include "CacheFlush.h"
#include "HorizontalDiffusionSA.h"
#include "HoriDiffReference.h"
#include "Definitions.h"
#include "UnittestEnvironment.h"
#include "HaloUpdateManager.h"

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

    // set up cache flusher for x86
    CacheFlush cacheFlusher(cCacheFlusherSize);

    // generate a reference field that contain the output of a horizontal diffusion operator
    HorizontalDiffusionSA horizontalDiffusionSA;
    horizontalDiffusionSA.Init(*pRepository_, UnittestEnvironment::getInstance().communicationConfiguration());

    HaloUpdateManager<true, false> haloUpdate;
    haloUpdate.Init("HaloUpdate", UnittestEnvironment::getInstance().communicationConfiguration());
    IJBoundary innerBoundary, outerBoundary;
    innerBoundary.Init(0, 0, 0, 0);
    outerBoundary.Init(-1, 1, -1, 1);
    haloUpdate.AddJob(pRepository_->u_out(0), innerBoundary, outerBoundary);

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
    for(int i=0; i < cNumBenchmarkRepetitions; ++i) {
        // flush cache between calls to horizontal diffusion stencil
        horizontalDiffusionSA.Apply();
        haloUpdate.Start();
        haloUpdate.Wait();
    }
#ifdef __CUDA_BACKEND__
    cudaProfilerStop();
#endif

//    std::cout << horizontalDiffusionSA.stencil().performanceMeter().ToString() << std::endl;
}



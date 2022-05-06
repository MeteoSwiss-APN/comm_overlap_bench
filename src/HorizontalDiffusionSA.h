#pragma once

#include "Definitions.h"
#include "Repository.h"
#include <vector>
#ifdef CUDA_BACKEND
#include <cuda_runtime.h>
#endif
#include <cassert>
#include <memory>

#define N_CONCURRENT_HALOS 2

/**
* @class HorizontalDiffusionSA
* Class holding the horizontal diffusion stencil for u and v
*/
class HorizontalDiffusionSA {
    DISALLOW_COPY_AND_ASSIGN(HorizontalDiffusionSA);

  public:
    HorizontalDiffusionSA(std::shared_ptr< Repository > repository);
    ~HorizontalDiffusionSA();

    /**
    * Method applying the u stencil
    */
    void Apply();
    void StartHalos(const int index);
    void WaitHalos(const int index);
    void ApplyHalos(const int i);

  private:
    const int commSize_;
    int cartSizes_[2];
    int neighbours_[4];
    MPI_Request requestNull;
    std::vector< MPI_Request > reqsIrecv_;
    std::vector< MPI_Request > reqsIsend_;
    MPI_Status status_[4 * N_CONCURRENT_HALOS];
    int numRanks_;
    int rankId_;

    std::vector< Real* > recWBuff_;
    std::vector< Real* > recNBuff_;
    std::vector< Real* > recEBuff_;
    std::vector< Real* > recSBuff_;

    std::vector< Real* > sendWBuff_;
    std::vector< Real* > sendNBuff_;
    std::vector< Real* > sendEBuff_;
    std::vector< Real* > sendSBuff_;

    std::shared_ptr< Repository > pRepository_;

#ifdef CUDA_BACKEND
    cudaStream_t kernelStream_;
#endif

    void fillRandom(SimpleStorage< Real >& storage);
    void generateFields(Repository& repository);
};

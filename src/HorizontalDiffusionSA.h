#pragma once

#include <vector>
#include "Definitions.h"
#include "Repository.h"
#include <cuda_runtime.h>
#include <cassert>
#include <memory>

#define N_CONCURRENT_HALOS 2

/**
* @class HorizontalDiffusionSA
* Class holding the horizontal diffusion stencil for u and v
*/
class HorizontalDiffusionSA
{
    DISALLOW_COPY_AND_ASSIGN(HorizontalDiffusionSA);
public:
    HorizontalDiffusionSA(std::shared_ptr<Repository> repository);
    ~HorizontalDiffusionSA();

    void Init(Repository* repo);

    /**
    * Method applying the u stencil
    */
    void Apply();

    void ResetMeters();

    void StartHalos(const int index)
    {

        cudaDeviceSynchronize();
        assert(index < N_CONCURRENT_HALOS);
        for(int c=0; c < N_HORIDIFF_VARS; ++c)
        {
            MPI_Irecv(recWBuff_[c*N_CONCURRENT_HALOS+index], commSize_, MPITYPE, neighbours_[0],5, MPI_COMM_WORLD, &(reqs_[c*4*N_CONCURRENT_HALOS+index]));
            MPI_Irecv(recNBuff_[c*N_CONCURRENT_HALOS+index], commSize_, MPITYPE, neighbours_[1],7, MPI_COMM_WORLD, &(reqs_[(c*4+1)*N_CONCURRENT_HALOS+index]));
            MPI_Irecv(recEBuff_[c*N_CONCURRENT_HALOS+index], commSize_, MPITYPE, neighbours_[2],1, MPI_COMM_WORLD, &(reqs_[(c*4+2)*N_CONCURRENT_HALOS+index]));
            MPI_Irecv(recSBuff_[c*N_CONCURRENT_HALOS+index], commSize_, MPITYPE, neighbours_[3],3, MPI_COMM_WORLD, &(reqs_[(c*4+3)*N_CONCURRENT_HALOS+index]));

            MPI_Isend(sendWBuff_[c*N_CONCURRENT_HALOS+index], commSize_, MPITYPE, neighbours_[0],1, MPI_COMM_WORLD, &requestNull);
            MPI_Isend(sendNBuff_[c*N_CONCURRENT_HALOS+index], commSize_, MPITYPE, neighbours_[1],3, MPI_COMM_WORLD, &requestNull);
            MPI_Isend(sendEBuff_[c*N_CONCURRENT_HALOS+index], commSize_, MPITYPE, neighbours_[2],5, MPI_COMM_WORLD, &requestNull);
            MPI_Isend(sendSBuff_[c*N_CONCURRENT_HALOS+index], commSize_, MPITYPE, neighbours_[3],7, MPI_COMM_WORLD, &requestNull);

#ifdef VERBOSE
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            std::cout << "Sending at " << rank << " with size " << commSize_ << " to " << neighbours_[0] << std::endl;
            std::cout << "Sending at " << rank << " with size " << commSize_ << " to " << neighbours_[1] << std::endl;
            std::cout << "Sending at " << rank << " with size " << commSize_ << " to " << neighbours_[2] << std::endl;
            std::cout << "Sending at " << rank << " with size " << commSize_ << " to " << neighbours_[3] << std::endl;
#endif
        }
    }

    void WaitHalos(const int index)
    {
        assert(index < N_CONCURRENT_HALOS);
        for(int c=0; c < N_HORIDIFF_VARS; ++c)
        {
            MPI_Wait(&(reqs_[c*4*N_CONCURRENT_HALOS+index]), &(status_[0*N_CONCURRENT_HALOS+index]));
            MPI_Wait(&(reqs_[(c*4+1)*N_CONCURRENT_HALOS+index]), &(status_[1*N_CONCURRENT_HALOS+index]));
            MPI_Wait(&(reqs_[(c*4+2)*N_CONCURRENT_HALOS+index]), &(status_[2*N_CONCURRENT_HALOS+index]));
            MPI_Wait(&(reqs_[(c*4+3)*N_CONCURRENT_HALOS+index]), &(status_[3*N_CONCURRENT_HALOS+index]));
        }

    }
    void ApplyHalos(const int i)
    {
        StartHalos(i);
        WaitHalos(i);
    }

private:
    int commSize_;
    int cartSizes_[2];
    int neighbours_[4];
    MPI_Request requestNull;
    MPI_Request *reqs_;
    MPI_Status status_[4*N_CONCURRENT_HALOS];
    int numRanks_;
    int rankId_;

    std::vector<Real*> recWBuff_;
    std::vector<Real*> recNBuff_;
    std::vector<Real*> recEBuff_;
    std::vector<Real*> recSBuff_;

    std::vector<Real*> sendWBuff_;
    std::vector<Real*> sendNBuff_;
    std::vector<Real*> sendEBuff_;
    std::vector<Real*> sendSBuff_;

    std::shared_ptr<Repository> pRepository_;
    cudaStream_t kernelStream_;
};

  

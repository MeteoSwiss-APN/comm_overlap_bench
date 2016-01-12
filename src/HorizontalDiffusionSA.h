#pragma once

//#include "HaloUpdateFramework.h"
#include "HoriDiffRepository.h"
#include "CommunicationConfiguration.h"
#include "Stencil.h"
#include "HaloUpdateManager.h"
#include "Options.h"

#define N_CONCURRENT_HALOS 2

/**
* @class HorizontalDiffusionSA
* Class holding the horizontal diffusion stencil for u and v
*/
class HorizontalDiffusionSA
{
    DISALLOW_COPY_AND_ASSIGN(HorizontalDiffusionSA);
public:
#ifdef __CUDA_BACKEND__
    //typedef BlockSize<64,4> HorizontalDiffusionUVBlockSize;
    typedef BlockSize<32,4> HorizontalDiffusionSABlockSize;
#else
    typedef BlockSize<4,4> HorizontalDiffusionSABlockSize;
#endif

    HorizontalDiffusionSA();
    ~HorizontalDiffusionSA();

    void Init(HoriDiffRepository& horiDiffRepository, CommunicationConfiguration& comm);

    /**
    * Method applying the u stencil
    */
    void Apply();

    void Apply(IJBoundary );

    void ResetMeters();

    void StartHalos(const int index)
    {
        assert(index < N_CONCURRENT_HALOS);
        for(int c=0; c < N_HORIDIFF_VARS; ++c)
        {
            if(Options::getInstance().nogcl_)
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
            else {

                assert(haloUpdates_[c*N_CONCURRENT_HALOS+index]);
                haloUpdates_[c*N_CONCURRENT_HALOS+index]->Start();
            }
        }
    }

    void WaitHalos(const int index)
    {
        assert(index < N_CONCURRENT_HALOS);
        for(int c=0; c < N_HORIDIFF_VARS; ++c)
        {
            if(Options::getInstance().nogcl_)
            {
                MPI_Wait(&(reqs_[c*4*N_CONCURRENT_HALOS+index]), &(status_[0*N_CONCURRENT_HALOS+index]));
                MPI_Wait(&(reqs_[(c*4+1)*N_CONCURRENT_HALOS+index]), &(status_[1*N_CONCURRENT_HALOS+index]));
                MPI_Wait(&(reqs_[(c*4+2)*N_CONCURRENT_HALOS+index]), &(status_[2*N_CONCURRENT_HALOS+index]));
                MPI_Wait(&(reqs_[(c*4+3)*N_CONCURRENT_HALOS+index]), &(status_[3*N_CONCURRENT_HALOS+index]));
            }
            else {

                assert(haloUpdates_[c*N_CONCURRENT_HALOS+index]);
                haloUpdates_[c*N_CONCURRENT_HALOS+index]->Wait();
            }
        }

    }
    void ApplyHalos(const int i)
    {
        StartHalos(i);
        WaitHalos(i);
    }

private:
    std::vector<Stencil*> stencils_;
    int commSize_;
    int cartSizes_[2];
    int neighbours_[4];
    MPI_Request requestNull;
    MPI_Request *reqs_;
    MPI_Status status_[4*N_CONCURRENT_HALOS];
    int numRanks_;
    int rankId_;

    std::vector<double*> recWBuff_;
    std::vector<double*> recNBuff_;
    std::vector<double*> recEBuff_;
    std::vector<double*> recSBuff_;

    std::vector<double*> sendWBuff_;
    std::vector<double*> sendNBuff_;
    std::vector<double*> sendEBuff_;
    std::vector<double*> sendSBuff_;

    std::vector<HaloUpdateManager<true, false>*> haloUpdates_;

    CommunicationConfiguration* pCommunicationConfiguration_;
    HoriDiffRepository *pHoriDiffRepository_;
    cudaStream_t kernelStream_;
};

  

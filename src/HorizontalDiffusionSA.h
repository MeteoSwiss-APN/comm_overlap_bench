#pragma once

//#include "HaloUpdateFramework.h"
#include "HoriDiffRepository.h"
#include "CommunicationConfiguration.h"
#include "Stencil.h"
#include "HaloUpdateManager.h"
#include "Options.h"

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

    void StartHalos()
    {
        for(int c=0; c < N_HORIDIFF_VARS; ++c)
        {
            if(Options::getInstance().nogcl_)
            {
                MPI_Isend(sendWBuff_[c], commSize_, MPITYPE, neighbours_[0],1, MPI_COMM_WORLD, &requestNull);
                MPI_Isend(sendNBuff_[c], commSize_, MPITYPE, neighbours_[1],1, MPI_COMM_WORLD, &requestNull);
                MPI_Isend(sendEBuff_[c], commSize_, MPITYPE, neighbours_[2],1, MPI_COMM_WORLD, &requestNull);
                MPI_Isend(sendSBuff_[c], commSize_, MPITYPE, neighbours_[3],1, MPI_COMM_WORLD, &requestNull);

                MPI_Irecv(recWBuff_[c], commSize_, MPITYPE, neighbours_[0],1, MPI_COMM_WORLD, &(reqs_[c*4]));
                MPI_Irecv(recNBuff_[c], commSize_, MPITYPE, neighbours_[1],1, MPI_COMM_WORLD, &(reqs_[c*4+1]));
                MPI_Irecv(recEBuff_[c], commSize_, MPITYPE, neighbours_[2],1, MPI_COMM_WORLD, &(reqs_[c*4+2]));
                MPI_Irecv(recSBuff_[c], commSize_, MPITYPE, neighbours_[3],1, MPI_COMM_WORLD, &(reqs_[c*4+3]));
            }
            else {

                assert(haloUpdates_[c]);
                haloUpdates_[c]->Start();
            }
        }
    }

    void WaitHalos()
    {
        for(int c=0; c < N_HORIDIFF_VARS; ++c)
        {
            if(Options::getInstance().nogcl_)
            {
                MPI_Wait(&(reqs_[c*4]), &(status_[0]));
                MPI_Wait(&(reqs_[c*4+1]), &(status_[1]));
                MPI_Wait(&(reqs_[c*4+2]), &(status_[2]));
                MPI_Wait(&(reqs_[c*4+3]), &(status_[3]));
            }
            else {

                assert(haloUpdates_[c]);
                haloUpdates_[c]->Wait();
            }
        }

    }
    void ApplyHalos()
    {
        StartHalos();
        WaitHalos();
    }

private:
    std::vector<Stencil*> stencils_;
    int commSize_;
    int cartSizes_[2];
    int neighbours_[4];
    MPI_Request requestNull;
    MPI_Request *reqs_;
    MPI_Status status_[4];
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
};

  

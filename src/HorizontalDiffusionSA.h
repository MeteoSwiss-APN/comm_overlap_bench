#pragma once

//#include "HaloUpdateFramework.h"
#include "HoriDiffRepository.h"
#include "CommunicationConfiguration.h"
#include "Stencil.h"
#include "HaloUpdateManager.h"

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
            assert(haloUpdates_[c]);
            haloUpdates_[c]->Start();
        }
    }

    void WaitHalos()
    {
        for(int c=0; c < N_HORIDIFF_VARS; ++c)
        {
            assert(haloUpdates_[c]);
            haloUpdates_[c]->Wait();
        }

    }
    void ApplyHalos()
    {
        StartHalos();
        WaitHalos();
    }

private:
    std::vector<Stencil*> stencils_;

    std::vector<HaloUpdateManager<true, false>*> haloUpdates_;

    CommunicationConfiguration* pCommunicationConfiguration_;
    HoriDiffRepository *pHoriDiffRepository_;
};

  

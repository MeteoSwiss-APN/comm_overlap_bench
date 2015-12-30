#pragma once

//#include "HaloUpdateFramework.h"
#include "HoriDiffRepository.h"
#include "CommunicationConfiguration.h"
#include "Stencil.h"

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


    void ResetMeters();
private:
    std::vector<Stencil*> stencils_;

    HoriDiffRepository *pHoriDiffRepository_;
};

  

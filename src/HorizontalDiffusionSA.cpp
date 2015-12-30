#include <boost/preprocessor/repetition/repeat.hpp>

#include "StencilFramework.h"
#include "MathFunctions.h"
#include "HorizontalDiffusionFunctions.h"
#include "HorizontalDiffusionSA.h"

// define parameter enum
enum
{
    u_in, u_out, lap, hdmaskvel, crlato, crlatu, crlavo, crlavu
};

namespace HorizontalDiffusionSAStages
{
    /**
    * @struct ULapStage
    * Corresponds to numeric_utilities.f90 - first loop of lap_4am
    */
    template<typename TEnv>
    struct ULapStage
    {
        STENCIL_STAGE(TEnv)

        STAGE_PARAMETER(FullDomain, u_out)
        STAGE_PARAMETER(FullDomain, u_in)
        STAGE_PARAMETER(FullDomain, crlato)
        STAGE_PARAMETER(FullDomain, crlatu)
        
        // flops: 6
        // accesses
        //    no cache: 2 (u, lap) . crlat are J fields not taken into account
        //    cache:    1+1/10
        __ACC__
        static void Do(Context ctx, FullDomain)
        {
            ctx[u_out::Center()] =
                ctx[Call<Laplacian>::With(u_in::Center(), crlato::Center(), crlatu::Center())];
        }
    };
}

HorizontalDiffusionSA::HorizontalDiffusionSA() : stencils_(N_HORIDIFF_VARS) 
{
    for(int c=0; c < N_HORIDIFF_VARS; ++c)
    {
        stencils_[c] = new Stencil();    
    }
        
}
HorizontalDiffusionSA::~HorizontalDiffusionSA() 
{
    for(int c=0; c < N_HORIDIFF_VARS; ++c)
    {
        assert(stencils_[c]);
        delete stencils_[c];
    }    
}

void HorizontalDiffusionSA::ResetMeters()
{
    for(int c=0; c < N_HORIDIFF_VARS; ++c)
    {
        assert(stencils_[c]);
        stencils_[c]->ResetMeters();
    }    

}
void HorizontalDiffusionSA::Init(
        HoriDiffRepository& horiDiffRepository, CommunicationConfiguration& comm)
{
    // store pointers to the repositories
    pHoriDiffRepository_ = &horiDiffRepository;

    using namespace HorizontalDiffusionSAStages;

    for(int c=0; c < N_HORIDIFF_VARS; ++c)
    {
        assert(stencils_[c]);
        // init the stencil u
        StencilCompiler::Build(
            *(stencils_[c]),
            "HorizontalDiffusion",
            horiDiffRepository.calculationDomain(),
            StencilConfiguration<Real, HorizontalDiffusionSABlockSize>(),
            pack_parameters(
                /* output fields */
                Param<u_out, cInOut, cDataField>(horiDiffRepository.u_out(c)),
                /* input fields */
                Param<u_in, cIn, cDataField>(horiDiffRepository.u_in(c)),
                Param<hdmaskvel, cIn, cDataField>(horiDiffRepository.hdmaskvel()),
                Param<crlato, cIn, cDataField>(horiDiffRepository.crlato()),
                Param<crlatu, cIn, cDataField>(horiDiffRepository.crlatu())
            ),
            define_loops(
                define_sweep<cKIncrement>(
                    define_stages(
                        StencilStage<ULapStage, IJRange<cIndented,0,0,0,0>, KRange<FullDomain,0,0> >()
                    )
                )
            )
        );
   }
}

void HorizontalDiffusionSA::Apply()
{
    for(int c=0; c < N_HORIDIFF_VARS; ++c)
    {
        assert(stencils_[c]);
        stencils_[c]->Apply();
    }
}


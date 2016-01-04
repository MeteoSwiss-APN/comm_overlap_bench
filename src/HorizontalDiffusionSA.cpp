#include <boost/preprocessor/repetition/repeat.hpp>

#include "StencilFramework.h"
#include "MathFunctions.h"
#include "HorizontalDiffusionFunctions.h"
#include "HorizontalDiffusionSA.h"
#include "Kernel.h"
#include "Options.h"

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

HorizontalDiffusionSA::HorizontalDiffusionSA() : stencils_(N_HORIDIFF_VARS), haloUpdates_(N_HORIDIFF_VARS)
{
    for(int c=0; c < N_HORIDIFF_VARS; ++c)
    {
        stencils_[c] = new Stencil();
        haloUpdates_[c]= new HaloUpdateManager<true, false>();
    }
        
}
HorizontalDiffusionSA::~HorizontalDiffusionSA() 
{
    for(int c=0; c < N_HORIDIFF_VARS; ++c)
    {
        assert(stencils_[c]);
        delete stencils_[c];
        assert(haloUpdates_[c]);
        delete haloUpdates_[c];
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

    pCommunicationConfiguration_ = &comm;
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

        assert(haloUpdates_[c]);
        assert(pCommunicationConfiguration_);
        haloUpdates_[c]->Init("HaloUpdate", *pCommunicationConfiguration_);
        IJBoundary innerBoundary, outerBoundary;
        innerBoundary.Init(0, 0, 0, 0);
        outerBoundary.Init(-3, 3, -3, 3);
        haloUpdates_[c]->AddJob(horiDiffRepository.u_out(c), innerBoundary, outerBoundary);

   }
}

void HorizontalDiffusionSA::Apply()
{
    for(int c=0; c < N_HORIDIFF_VARS; ++c)
    {
        if(Options::getInstance().nostella_)
        {
            launch_kernel(
                        pHoriDiffRepository_->calculationDomain(),
                        pHoriDiffRepository_->u_in(c).storage().pStorageBase(),
                        pHoriDiffRepository_->u_out(c).storage().pStorageBase());
        }
        else
        {
            assert(stencils_[c]);
            stencils_[c]->Apply();
        }
    }
}

void HorizontalDiffusionSA::Apply(IJBoundary ijBoundary)
{
    for(int c=0; c < N_HORIDIFF_VARS; ++c)
    {
        assert(stencils_[c]);
        stencils_[c]->Apply(ijBoundary);
    }
}

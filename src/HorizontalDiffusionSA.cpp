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

HorizontalDiffusionSA::HorizontalDiffusionSA() : stencils_(N_HORIDIFF_VARS), haloUpdates_(N_HORIDIFF_VARS),
    recWBuff_(N_HORIDIFF_VARS), recNBuff_(N_HORIDIFF_VARS), recEBuff_(N_HORIDIFF_VARS), recSBuff_(N_HORIDIFF_VARS),
    sendWBuff_(N_HORIDIFF_VARS), sendNBuff_(N_HORIDIFF_VARS), sendEBuff_(N_HORIDIFF_VARS), sendSBuff_(N_HORIDIFF_VARS)
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

    MPI_Comm_size(MPI_COMM_WORLD, &numRanks_);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId_);

    bool found=false;
    for(int i=sqrt(numRanks_); i > 0; --i)
    {
        if(numRanks_ % i == 0) {
            found = true;
            cartSizes_[0] = i;
            cartSizes_[1] = numRanks_ / i;
            break;
        }
    }
    if(!found) std::cout<< "ERROR: Could not apply a domain decomposition" << std::endl;

    if(rankId_ % cartSizes_[0] == 0)
        neighbours_[0] = rankId_ + cartSizes_[0]-1;
    else
        neighbours_[0] = rankId_ -1;

    if(rankId_ < cartSizes_[0])
        neighbours_[1] = rankId_ + (cartSizes_[1]-1)*cartSizes_[0];
    else
        neighbours_[1] = rankId_ - cartSizes_[0];

    if((rankId_+1) % cartSizes_[0] == 0)
        neighbours_[2] = rankId_ - (cartSizes_[0]-1);
    else
        neighbours_[2] = rankId_+1;

    if(rankId_ >= (cartSizes_[1]-1)*cartSizes_[0] )
        neighbours_[3] = (rankId_ + cartSizes_[0] ) - cartSizes_[0]*cartSizes_[1];
    else
        neighbours_[3] =rankId_ + cartSizes_[0];

    commSize_ = (horiDiffRepository.calculationDomain().iSize()+cNumBoundaryLines*2)*3;

    for(int c=0; c < N_HORIDIFF_VARS; ++c)
    {
        cudaMalloc(&(sendWBuff_[c]), sizeof(Real)*commSize_);
        cudaMalloc(&(sendNBuff_[c]), sizeof(Real)*commSize_);
        cudaMalloc(&(sendEBuff_[c]), sizeof(Real)*commSize_);
        cudaMalloc(&(sendSBuff_[c]), sizeof(Real)*commSize_);

        cudaMalloc(&(recWBuff_[c]), sizeof(Real)*commSize_);
        cudaMalloc(&(recNBuff_[c]), sizeof(Real)*commSize_);
        cudaMalloc(&(recEBuff_[c]), sizeof(Real)*commSize_);
        cudaMalloc(&(recSBuff_[c]), sizeof(Real)*commSize_);
    }

    reqs_ = (MPI_Request*)malloc((N_HORIDIFF_VARS)*sizeof(MPI_Request)*4);

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

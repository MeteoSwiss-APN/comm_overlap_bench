#include <boost/preprocessor/repetition/repeat.hpp>

#include "HorizontalDiffusionSA.h"
#include "Kernel.h"

HorizontalDiffusionSA::HorizontalDiffusionSA(std::shared_ptr<Repository> repository):
  recWBuff_(N_HORIDIFF_VARS*N_CONCURRENT_HALOS),
  recNBuff_(N_HORIDIFF_VARS*N_CONCURRENT_HALOS),
  recEBuff_(N_HORIDIFF_VARS*N_CONCURRENT_HALOS),
  recSBuff_(N_HORIDIFF_VARS*N_CONCURRENT_HALOS),
  sendWBuff_(N_HORIDIFF_VARS*N_CONCURRENT_HALOS),
  sendNBuff_(N_HORIDIFF_VARS*N_CONCURRENT_HALOS),
  sendEBuff_(N_HORIDIFF_VARS*N_CONCURRENT_HALOS),
  sendSBuff_(N_HORIDIFF_VARS*N_CONCURRENT_HALOS),
  pRepository_(repository)
{
    cudaStreamCreate(&kernelStream_);

    const IJKSize& domain = repository->domain;
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

    commSize_ = domain.isize*3*domain.ksize;

    for(int c=0; c < N_HORIDIFF_VARS; ++c)
    {
        for(int h=0; h < N_CONCURRENT_HALOS; ++h)
        {
            cudaMalloc(&(sendWBuff_[c*N_CONCURRENT_HALOS+h]), sizeof(Real)*commSize_);
            cudaMalloc(&(sendNBuff_[c*N_CONCURRENT_HALOS+h]), sizeof(Real)*commSize_);
            cudaMalloc(&(sendEBuff_[c*N_CONCURRENT_HALOS+h]), sizeof(Real)*commSize_);
            cudaMalloc(&(sendSBuff_[c*N_CONCURRENT_HALOS+h]), sizeof(Real)*commSize_);

            cudaMalloc(&(recWBuff_[c*N_CONCURRENT_HALOS+h]), sizeof(Real)*commSize_);
            cudaMalloc(&(recNBuff_[c*N_CONCURRENT_HALOS+h]), sizeof(Real)*commSize_);
            cudaMalloc(&(recEBuff_[c*N_CONCURRENT_HALOS+h]), sizeof(Real)*commSize_);
            cudaMalloc(&(recSBuff_[c*N_CONCURRENT_HALOS+h]), sizeof(Real)*commSize_);
        }
    }

    reqs_ = (MPI_Request*)malloc(N_HORIDIFF_VARS*N_CONCURRENT_HALOS*sizeof(MPI_Request)*4);
}
HorizontalDiffusionSA::~HorizontalDiffusionSA() 
{
    cudaStreamDestroy(kernelStream_);

}

void HorizontalDiffusionSA::Apply()
{
    for(int c=0; c < N_HORIDIFF_VARS; ++c)
    {
          launch_kernel(pRepository_->domain,
                        pRepository_->in(c).device,
                        pRepository_->out(c).device,
                        kernelStream_
          );
    }
}


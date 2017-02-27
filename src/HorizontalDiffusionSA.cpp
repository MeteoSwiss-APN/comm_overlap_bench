#include <boost/preprocessor/repetition/repeat.hpp>
#include <exception>
#include "HorizontalDiffusionSA.h"
#include "Kernel.h"
#include <random>

HorizontalDiffusionSA::HorizontalDiffusionSA(std::shared_ptr<Repository> repository):
  commSize_(repository->domain.isizeFull()*cNumBoundaryLines*repository->domain.ksizeFull()),
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
    generateFields(*pRepository_);
    cudaStreamCreate(&kernelStream_);

    const IJKSize& domain = repository->domain;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks_);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId_);

    std::vector<int> periods {1, 1};
    std::vector<int> dims {0, 0};
    MPI_Comm cartcomm;
    MPI_Dims_create(numRanks_, 2, &dims[0]);
    if (dims[0]*dims[1] != numRanks_) {
        throw std::runtime_error("Can't create 2D cartesian communicator: Invalid number of ranks");
    }
    MPI_Cart_create(MPI_COMM_WORLD, 2, &dims[0], &periods[0], 1, &cartcomm);

    std::vector<int> cartPos {-1, -1};
    MPI_Cart_coords(cartcomm, rankId_, 2, &cartPos[0]);

    std::vector<int> coords {0, 0};

    coords[1] = cartPos[1];
    if (cartPos[0] == 0) {
        coords[0] = dims[0]-1;
    } else {
        coords[0] = cartPos[0]-1;
    }
    MPI_Cart_rank(cartcomm, &(coords[0]), &(neighbours_[0]));

    coords[0] = cartPos[0];
    if (cartPos[1] == dims[1]-1) {
        coords[1] = 0;
    } else {
        coords[1] = cartPos[1]+1;
    }
    MPI_Cart_rank(cartcomm,  &(coords[0]), &neighbours_[1]);

    coords[1] = cartPos[1];
    if (cartPos[0] == dims[1]-1) {
        coords[0] = 0;
    } else {
        coords[0] = coords[0]+1;
    }
    MPI_Cart_rank(cartcomm,  &(coords[0]), &neighbours_[2]);

    coords[0] = cartPos[0];
    if (cartPos[1] == 0) {
        coords[1] = dims[1]-1;
    } else {
        coords[1] = cartPos[1]-1;
    }
    MPI_Cart_rank(cartcomm,  &(coords[0]), &neighbours_[3]);

    std::cout << "MPI Configuration for rank " << std::to_string(rankId_) << "\n";
    std::cout << "Dimensions: [" << std::to_string(dims[0]) << ", " << std::to_string(dims[1]) << "]\n";
    std::cout << "Neighbors: [" << std::to_string(neighbours_[0]) << ", "  << std::to_string(neighbours_[1]) << ", "  << std::to_string(neighbours_[2]) << ", "  << std::to_string(neighbours_[3]) << "]\n";

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

void HorizontalDiffusionSA::fillRandom(SimpleStorage<Real> &storage)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distribution(0,1);
    const IJKSize domain = storage.size;
    for (int i=0; i<domain.isizeFull(); ++i) {
        for (int j=0; j<domain.jsizeFull(); ++j) {
            for (int k=0; k<domain.ksizeFull(); ++k) {
                storage(i,j,k) = distribution(gen);
            }
        }
    }
    storage.updateDevice();
}

void HorizontalDiffusionSA::generateFields(Repository& repository)
{
    for (size_t i=0; i<N_HORIDIFF_VARS; ++i) {
        fillRandom(repository.in(i));
        fillRandom(repository.out(i));
    }
}

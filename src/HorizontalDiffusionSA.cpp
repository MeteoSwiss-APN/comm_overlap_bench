#ifndef _NO_BOOST
#include <boost/preprocessor/repetition/repeat.hpp>
#endif
#include <exception>
#include "HorizontalDiffusionSA.h"

#ifdef CUDA_BACKEND
#include "Kernel.h"
#endif
#include "MPIHelper.h"
#include <random>
#include <iostream>

HorizontalDiffusionSA::HorizontalDiffusionSA(std::shared_ptr<Repository> repository):
  commSize_(repository->domain.isizeFull()*cNumBoundaryLines*repository->domain.ksizeFull()),
  reqsIrecv_(N_HORIDIFF_VARS*N_CONCURRENT_HALOS*4),
  reqsIsend_(N_HORIDIFF_VARS*N_CONCURRENT_HALOS*4),
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
#ifdef CUDA_BACKEND
    cudaStreamCreate(&kernelStream_);
#endif

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
    const int& cX = cartPos[0];
    const int& cY = cartPos[1];
    int& nX = coords[0];
    int& nY = coords[1];

    const int& dimX = dims[0];
    const int& dimY = dims[1];

    nY = cY;
    if (cX == 0) {
        nX = dimX-1;
    } else {
        nX = cX-1;
    }
    MPI_Cart_rank(cartcomm, &nX, &(neighbours_[0]));

    nX = cX;
    if (cY == dimY-1) {
        nY = 0;
    } else {
        nY = cY+1;
    }
    MPI_Cart_rank(cartcomm,  &nX, &neighbours_[1]);

    nY = cY;
    if (cX == dimX-1) {
        nX = 0;
    } else {
        nX = cX+1;
    }
    MPI_Cart_rank(cartcomm,  &nX, &neighbours_[2]);

    nX = cX;
    if (cY == 0) {
        nY = dimY-1;
    } else {
        nY = cY-1;
    }
    MPI_Cart_rank(cartcomm,  &nX, &neighbours_[3]);

    if (rankId_ == 0) {
        std::cout << "Dimensions: [" << std::to_string(dims[0]) << ", " << std::to_string(dims[1]) << "]" << std::endl;
    }
    MPIHelper::print("Neighbors: ", "["+std::to_string(rankId_)+": "
                                           +std::to_string(neighbours_[0])+","
                                           +std::to_string(neighbours_[1])+","
                                           +std::to_string(neighbours_[2])+","
                                           +std::to_string(neighbours_[3])+"] ", 9999);

#ifdef CUDA_BACKEND
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
#else

    for(int c=0; c < N_HORIDIFF_VARS; ++c)
    {
        for(int h=0; h < N_CONCURRENT_HALOS; ++h)
        {
            sendWBuff_[c*N_CONCURRENT_HALOS+h] = (Real*) malloc(sizeof(Real)*commSize_);
            sendNBuff_[c*N_CONCURRENT_HALOS+h] = (Real*) malloc(sizeof(Real)*commSize_);
            sendEBuff_[c*N_CONCURRENT_HALOS+h] = (Real*) malloc(sizeof(Real)*commSize_);
            sendSBuff_[c*N_CONCURRENT_HALOS+h] = (Real*) malloc(sizeof(Real)*commSize_);

            recWBuff_[c*N_CONCURRENT_HALOS+h] = (Real*) malloc(sizeof(Real)*commSize_);
            recNBuff_[c*N_CONCURRENT_HALOS+h] = (Real*) malloc(sizeof(Real)*commSize_);
            recEBuff_[c*N_CONCURRENT_HALOS+h] = (Real*) malloc(sizeof(Real)*commSize_);
            recSBuff_[c*N_CONCURRENT_HALOS+h] = (Real*) malloc(sizeof(Real)*commSize_);
        }
    }

    std::cout << "Warning: CUDA not enabled" << std::endl;
#endif
}
HorizontalDiffusionSA::~HorizontalDiffusionSA() 
{
#ifdef CUDA_BACKEND
    cudaStreamDestroy(kernelStream_);
#endif
}


void HorizontalDiffusionSA::StartHalos(const int index)
{
#ifdef CUDA_BACKEND
    cudaDeviceSynchronize();
#endif
    assert(index < N_CONCURRENT_HALOS);
    for(int c=0; c < N_HORIDIFF_VARS; ++c)
    {
        MPI_Irecv(recWBuff_[c*N_CONCURRENT_HALOS+index], commSize_, MPITYPE, neighbours_[0],5, MPI_COMM_WORLD, &(reqsIrecv_[(c*4+0)*N_CONCURRENT_HALOS+index]));
        MPI_Irecv(recNBuff_[c*N_CONCURRENT_HALOS+index], commSize_, MPITYPE, neighbours_[1],7, MPI_COMM_WORLD, &(reqsIrecv_[(c*4+1)*N_CONCURRENT_HALOS+index]));
        MPI_Irecv(recEBuff_[c*N_CONCURRENT_HALOS+index], commSize_, MPITYPE, neighbours_[2],1, MPI_COMM_WORLD, &(reqsIrecv_[(c*4+2)*N_CONCURRENT_HALOS+index]));
        MPI_Irecv(recSBuff_[c*N_CONCURRENT_HALOS+index], commSize_, MPITYPE, neighbours_[3],3, MPI_COMM_WORLD, &(reqsIrecv_[(c*4+3)*N_CONCURRENT_HALOS+index]));

        MPI_Isend(sendWBuff_[c*N_CONCURRENT_HALOS+index], commSize_, MPITYPE, neighbours_[0],1, MPI_COMM_WORLD, &(reqsIsend_[(c*4+0)*N_CONCURRENT_HALOS+index]));
        MPI_Isend(sendNBuff_[c*N_CONCURRENT_HALOS+index], commSize_, MPITYPE, neighbours_[1],3, MPI_COMM_WORLD, &(reqsIsend_[(c*4+1)*N_CONCURRENT_HALOS+index]));
        MPI_Isend(sendEBuff_[c*N_CONCURRENT_HALOS+index], commSize_, MPITYPE, neighbours_[2],5, MPI_COMM_WORLD, &(reqsIsend_[(c*4+2)*N_CONCURRENT_HALOS+index]));
        MPI_Isend(sendSBuff_[c*N_CONCURRENT_HALOS+index], commSize_, MPITYPE, neighbours_[3],7, MPI_COMM_WORLD, &(reqsIsend_[(c*4+3)*N_CONCURRENT_HALOS+index]));

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

void HorizontalDiffusionSA::WaitHalos(const int index)
{
    assert(index < N_CONCURRENT_HALOS);
    for(int c=0; c < N_HORIDIFF_VARS; ++c)
    {
        MPI_Wait(&(reqsIrecv_[c*4*N_CONCURRENT_HALOS+index]), &(status_[0*N_CONCURRENT_HALOS+index]));
        MPI_Wait(&(reqsIrecv_[(c*4+1)*N_CONCURRENT_HALOS+index]), &(status_[1*N_CONCURRENT_HALOS+index]));
        MPI_Wait(&(reqsIrecv_[(c*4+2)*N_CONCURRENT_HALOS+index]), &(status_[2*N_CONCURRENT_HALOS+index]));
        MPI_Wait(&(reqsIrecv_[(c*4+3)*N_CONCURRENT_HALOS+index]), &(status_[3*N_CONCURRENT_HALOS+index]));

        MPI_Request_free(&(reqsIsend_[(c*4+0)*N_CONCURRENT_HALOS+index]));
        MPI_Request_free(&(reqsIsend_[(c*4+1)*N_CONCURRENT_HALOS+index]));
        MPI_Request_free(&(reqsIsend_[(c*4+2)*N_CONCURRENT_HALOS+index]));
        MPI_Request_free(&(reqsIsend_[(c*4+3)*N_CONCURRENT_HALOS+index]));
    }

}
void HorizontalDiffusionSA::ApplyHalos(const int i)
{
    StartHalos(i);
    WaitHalos(i);
}



void HorizontalDiffusionSA::Apply()
{
#ifdef CUDA_BACKEND
    for(int c=0; c < N_HORIDIFF_VARS; ++c)
    {
          launch_kernel(pRepository_->domain,
                        pRepository_->in(c).device,
                        pRepository_->out(c).device,
                        kernelStream_
          );
    }
#endif
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

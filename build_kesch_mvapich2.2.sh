#!/usr/bin/env bash
rm -rf build_kesch_mvapich22
mkdir -p build_kesch_mvapich22

module purge
module load craype-haswell
module load craype-network-infiniband
module load mvapich2gdr_gnu/2.2_cuda_7.0
module load GCC/4.9.3-binutils-2.25
module load cray-libsci_acc/3.3.0

module load CMake
module list -t
echo

export CC=gcc
export CXX=g++
export BOOST_ROOT="/apps/escha/UES/RH6.7/easybuild/software/Boost/1.49.0-gmvolf-15.11-Python-2.7.10"

pushd build_kesch_mvapich22 &>/dev/null
    cmake .. \
             -DMPI_VENDOR=mvapich2 \
             -DCUDA_COMPUTE_CAPABILITY="sm_37" \
             -DCMAKE_BUILD_TYPE=Release \
             -DENABLE_BOOST_TIMER=ON \
             -DBOOST_ROOT="${BOOST_ROOT}" 
    make -j 1


#!/usr/bin/env bash

rm -rf build_cuda80
mkdir -p build_cuda80

module use /apps/escha/UES/RH6.8_PE17.02/easybuild/modules/all
module use /apps/common/UES/RHAT6/easybuild/modules/all
module purge
module load craype-haswell
module load craype-network-infiniband
module load Score-P/3.0-gmvapich2-17.01_cuda_8.0_gdr
module load CMake

# Currently Loaded Modulefiles:
# craype-haswell
# Vim/8.0
# binutils/.2.25
# GCC/4.9.3-binutils-2.25
# cudatoolkit/.8.0.44   <--------
# mvapich2gdr_gnu/.2.2_cuda_8.0
# MVAPICH2/2.2-GCC-4.9.3-binutils-2.25_cuda_8.0_gdr
# gmvapich2/17.01_cuda_8.0_gdr
# libunwind/.1.1
# Cube/.4.3.4
# papi/5.4.3.2
# vampir/9.2.0
# Score-P/3.0-gmvapich2-17.01_cuda_8.0_gdr
# CMake/3.7.2

module list -t ;echo
mpiname -a ;echo
gcc -v 2>&1 |grep "gcc version";echo 
nvcc -V |grep "Cuda compilation tools" ;echo
nvidia-smi |grep NVIDIA-SMI ;echo

export CC=mpicc
export CXX=mpicxx
# export CC=gcc
# export CXX=g++
# export BOOST_ROOT="/apps/escha/UES/RH6.8_PE17.02/easybuild/software/Boost/1.63.0-gmvapich2-17.01_cuda_8.0_gdr-Python-2.7.12"

cd build_cuda80/

cmake .. \
-DMPI_VENDOR=mvapich2 \
-DCUDA_COMPUTE_CAPABILITY="sm_37" \
-DCMAKE_BUILD_TYPE=Release

make VERBOSE=1

# exit 0

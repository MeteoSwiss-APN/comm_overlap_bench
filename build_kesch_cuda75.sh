#!/usr/bin/env bash
rm -rf build_cuda75
mkdir -p build_cuda75

module purge
source modules_kesch_cuda75.env
module load CMake
module list -t
echo

export CC=gcc
export CXX=g++
export BOOST_ROOT="/apps/escha/UES/RH6.7/easybuild/software/Boost/1.63.0-gmvapich2-17.02_cuda_7.5_gdr-Python-2.7.12"

pushd build_cuda75 &>/dev/null
    cmake .. \
             -DMPI_VENDOR=mvapich2 \
             -DCUDA_COMPUTE_CAPABILITY="sm_37" \
             -DCMAKE_BUILD_TYPE=Release \
             -DENABLE_BOOST_TIMER=ON \
             -DBOOST_ROOT="${BOOST_ROOT}" 
    make -j 1 


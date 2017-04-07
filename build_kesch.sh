#!/usr/bin/env bash
rm -rf build
mkdir -p build

module purge
module load CMake
source modules_kesch.env
module list -t
echo

export CC=gcc
export CXX=g++
export BOOST_ROOT="/apps/escha/UES/RH6.7/easybuild/software/Boost/1.49.0-gmvolf-15.11-Python-2.7.10"

pushd build &>/dev/null
    cmake .. -DCMAKE_CXX_FLAGS="-std=c++11" \
             -DMPI_VENDOR=mvapich2 \
             -DCUDA_COMPUTE_CAPABILITY="sm_37" \
             -DCMAKE_BUILD_TYPE=Release \
             -DENABLE_TIMER=ON \
             -DBOOST_ROOT="${BOOST_ROOT}" 
    make -j 1


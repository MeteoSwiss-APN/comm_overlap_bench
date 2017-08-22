#!/usr/bin/env bash
rm -rf build
mkdir -p build

module purge
source modules_kesch.env
module load CMake
module list -t
echo

export CC=gcc
export CXX=g++
export BOOST_ROOT="/apps/escha/UES/RH6.7/easybuild/software/Boost/1.49.0-gmvolf-15.11-Python-2.7.10"

pushd build &>/dev/null
    cmake .. \
             -DMPI_VENDOR=mvapich2 \
             -DCUDA_COMPUTE_CAPABILITY="sm_37" \
             -DCMAKE_BUILD_TYPE=Release \
             -DENABLE_BOOST_TIMER=ON \
             -DBOOST_ROOT="${BOOST_ROOT}" 
    make -j 1


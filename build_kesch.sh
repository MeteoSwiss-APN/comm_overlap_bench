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

pushd build &>/dev/null
    cmake .. \
             -DMPI_VENDOR=mvapich2 \
             -DCUDA_COMPUTE_CAPABILITY="sm_37" \
             -DCMAKE_BUILD_TYPE=Release \
             -DENABLE_MPI_TIMER=ON
    make -j 1


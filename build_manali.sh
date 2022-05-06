#!/usr/bin/env bash
rm -rf build
mkdir -p build

module use /apps/manali/UES/jenkins/MCH-PE20.08-UP01/NVHPC/22.3/easybuild/modules/all
module load OpenMPI

module list -t
echo

export CC=gcc
export CXX=g++

pushd build &>/dev/null
    cmake .. \
             -DMPI_VENDOR=openmpi \
             -DCUDA_COMPUTE_CAPABILITY="sm_80" \
             -DCMAKE_BUILD_TYPE=Release \
             -DENABLE_MPI_TIMER=ON \
             -DVERBOSE=OFF
    make -j 4 VERBOSE=1


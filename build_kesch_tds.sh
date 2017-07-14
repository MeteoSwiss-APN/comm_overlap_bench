#!/usr/bin/env bash
rm -rf build
mkdir -p build

module purge
module load craype-network-infiniband
module load craype-haswell
module load git/2.6.0
module load tmux/2.1
module load cudatoolkit/8.0.61
module load mvapich2gdr_gnu/2.2_cuda_8.0
module load gcc/5.4.0-2.26
#module load gcc/4.8.1
module load cmake/3.7.2
module list -t
echo

#export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
export CC=gcc
export CXX=g++
#alias mpicc="/opt/mvapich2/gdr/no-mcast/2.2/cuda8.0/mpirun/gnu4.8.5/bin/mpicc `pkg-config --libs cudart`"
#alias mpicxx="/opt/mvapich2/gdr/no-mcast/2.2/cuda8.0/mpirun/gnu4.8.5/bin/mpicxx `pkg-config --libs cudart`"
export LDFLAGS=`pkg-config --libs cudart`
export BOOST_ROOT="/apps/escha/UES/RH6.7/easybuild/software/Boost/1.49.0-gmvolf-15.11-Python-2.7.10"

pushd build &>/dev/null
    cmake .. \
             -DMPI_VENDOR=mvapich2 \
             -DCUDA_COMPUTE_CAPABILITY="sm_37" \
             -DCMAKE_BUILD_TYPE=Release \
             -DENABLE_MPI_TIMER=ON
#             -DBOOST_ROOT="${BOOST_ROOT}" 
    make -j 1

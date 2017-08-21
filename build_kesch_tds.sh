#!/usr/bin/env bash
rm -rf build_tds
mkdir -p build_tds

module purge
module load craype-network-infiniband
module load craype-haswell
module load craype-accel-nvidia35
module load cray-libsci
module load git/2.6.0
module load tmux/2.1
module load cudatoolkit/8.0.61
module load mvapich2gdr_gnu/2.2_cuda_8.0
module load gcc/5.4.0-2.26
#module load gcc/4.8.1
module load cmake/3.7.2

module list -t
echo

export CC=gcc
export CXX=g++

export LDFLAGS=`pkg-config --libs cudart`
export BOOST_ROOT="/apps/escha/UES/jenkins/RH7.3-17.02/easybuild/software/boost/1.63.0-gmvolf-17.02-python-2.7.13/"

pushd build_tds &>/dev/null
    cmake .. \
             -DMPI_VENDOR=mvapich2 \
             -DCUDA_COMPUTE_CAPABILITY="sm_37" \
             -DCMAKE_BUILD_TYPE=Release \
             -DENABLE_TIMER=ON \
             -DBOOST_ROOT="${BOOST_ROOT}" 
    make -j 1

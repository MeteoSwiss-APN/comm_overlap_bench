#!/usr/bin/env bash
rm -rf build
mkdir -p build

module purge
module load CMake
source modules_kesch.env
module load Score-P/3.0-gmvapich2-15.11_cuda_7.0_gdr
module list -t
echo

export CC=gcc
export CXX=g++
#export CC=`which scorep-gcc`
#export CXX=`which scorep-g++`
#export NVCC=`which scorep-nvcc`
export CC=$(pwd)/cc.scorep
export CXX=$(pwd)/CC.scorep
export NVCC=$(pwd)/nvcc.scorep

export SCOREP_ROOT=$EBROOTSCOREMINP
export SCOREP_WRAPPER_ARGS="--mpp=mpi --cuda"

export BOOST_ROOT="/apps/escha/UES/RH6.7/easybuild/software/Boost/1.49.0-gmvolf-15.11-Python-2.7.10"

pushd build &>/dev/null
    SCOREP_WRAPPER=OFF cmake .. \
        -DCMAKE_CXX_FLAGS="-std=c++11" \
        -DMPI_VENDOR=mvapich2 \
        -DENABLE_TIMER=OFF \
        -DCUDA_COMPUTE_CAPABILITY="sm_37" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_COMPILER="${CXX}" \
        -DCMAKE_C_COMPILER="${CC}" \
        -DCUDA_HOST_COMPILER=`which g++` \
        -DCUDA_NVCC_EXECUTABLE="${NVCC}"

    # Only needed when the timers are enabled
    #-DBOOST_ROOT="${BOOST_ROOT}" \

    export SCOREP_WRAPPER=ON
    make -j 1 \
        SCOREP_WRAPPER_INSTRUMENTER_FLAGS="${SCOREP_WRAPPER_ARGS}" \
        VERBOSE=1


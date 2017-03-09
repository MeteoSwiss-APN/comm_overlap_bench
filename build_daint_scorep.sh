#!/usr/bin/env bash
rm -rf build_scorep
mkdir -p build_scorep

source modules_daint.sh
module load Score-P/3.0-CrayGNU-2016.11-cuda-8.0.54
module list -t
echo

export SCOREP_WRAPPER_ARGS="--mpp=mpi --cuda --keep-files --verbose"
export CC=`which scorep-cc`
export CXX=`which scorep-CC`
export NVCC=`which scorep-nvcc`

export BOOST_ROOT="/apps/daint/UES/jenkins/6.0.UP02/gpu/easybuild/software/Boost/1.63.0-CrayGNU-2016.11-Python-3.5.2/"

pushd build_scorep &>/dev/null
#    cmake .. -DCMAKE_CXX_FLAGS="-std=c++11" -DBOOST_ROOT="${BOOST_ROOT}" -DCUDA_COMPUTE_CAPABILITY="sm_60" -DCMAKE_BUILD_TYPE=Debug
SCOREP_WRAPPER=OFF cmake .. \
    -DCMAKE_CXX_FLAGS="-std=c++11" \
    -DENABLE_TIMER=OFF \
    -DCUDA_COMPUTE_CAPABILITY="sm_60" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBOOST_ROOT="${BOOST_ROOT}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCUDA_NVCC_EXECUTABLE="${NVCC}" \
    -DCUDA_HOST_COMPILER=`which CC`

    export SCOREP_WRAPPER=ON
    make -j 1 \
        SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--mpp=mpi --cuda --keep-files" 
#        VERBOSE=1


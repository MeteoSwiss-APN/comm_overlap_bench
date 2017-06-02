#!/usr/bin/env bash
rm -rf build
mkdir -p build

module purge
source modules_kesch.env
module use /users/piccinal/easybuild/keschln/modules/all
module load
module load Score-P/3.0-gmvapich2-15.11_cuda_7.0_gdr CMake/3.8.1
module list -t
echo

RRR=$(pwd)
export CC=$RRR/cc.scorep
export CXX=$RRR/mpicxx.scorep
export NVCC=$RRR/nvcc.scorep
export SCOREP_WRAPPER=OFF 

#export SCOREP_ROOT=$EBROOTSCOREMINP
#export SCOREP_WRAPPER_ARGS="--mpp=mpi --cuda"

export BOOST_ROOT="/apps/escha/UES/RH6.7/easybuild/software/Boost/1.49.0-gmvolf-15.11-Python-2.7.10"

pushd build &>/dev/null
    SCOREP_WRAPPER=OFF cmake .. \
        -DCMAKE_CXX_FLAGS="-std=c++11" \
        -DENABLE_TIMER=OFF \
        -DENABLE_MPI_TIMER=OFF \
        -DENABLE_SCOREP_TIMER=OFF \
        -DCUDA_COMPUTE_CAPABILITY="sm_37" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_COMPILER="${CXX}" \
        -DCMAKE_C_COMPILER="${CC}" \
        -DCUDA_HOST_COMPILER=`which g++` \
        -DCUDA_NVCC_EXECUTABLE="${NVCC}" \
        -DCMAKE_EXE_LINKER_FLAGS="-lpthread "
    # Only needed when the timers are enabled
    #-DBOOST_ROOT="${BOOST_ROOT}" \

    # if ENABLE_SCOREP_TIMER=ON we need to supply --user with the wrapper
    export SCOREP_WRAPPER=ON
    make -j 1 \
        VERBOSE=1


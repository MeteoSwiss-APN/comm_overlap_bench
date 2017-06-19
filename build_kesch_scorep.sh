#!/usr/bin/env bash
rm -rf build
mkdir -p build

# source modules_kesch.env
module purge
module load CMake
module load craype-haswell
module load craype-network-infiniband
# module load Boost/1.49.0-gmvolf-15.11-Python-2.7.10
# module unload MVAPICH2/2.2a-GCC-4.9.3-binutils-2.25
# module unload gmvapich2/15.11
# module load mvapich2gdr_gnu/2.1_cuda_7.0
# module load GCC/4.9.3-binutils-2.25
# module load cray-libsci_acc/3.3.0
#30+cuda70: module load Score-P/3.0-gmvapich2-15.11_cuda_7.0_gdr
#30+cuda75: 
module load Score-P/3.0-gmvapich2-17.02_cuda_7.5_gdr
#31: module use /users/piccinal/easybuild/keschln/modules/all
#31: module load Score-P/3.1-gmvapich2-17.02_cuda_7.5_gdr
module list -t
echo

RRR=$(pwd)
export CC=$RRR/cc.scorep
export CXX=$RRR/CXX.scorep
export NVCC=$RRR/nvcc.scorep
export SCOREP_WRAPPER=OFF 

#export SCOREP_ROOT=$EBROOTSCOREMINP
#export SCOREP_WRAPPER_ARGS="--mpp=mpi --cuda"
#export BOOST_ROOT="/apps/escha/UES/RH6.7/easybuild/software/Boost/1.49.0-gmvolf-15.11-Python-2.7.10"

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
        -DCMAKE_EXE_LINKER_FLAGS="-lpthread"
# Only needed when the timers are enabled
# -DBOOST_ROOT="${BOOST_ROOT}" \

# already set in *.scorep:
# SCOREP_WRAPPER_INSTRUMENTER_FLAGS="${SCOREP_WRAPPER_ARGS}" \
export SCOREP_WRAPPER=ON
make -j 1 \
VERBOSE=1


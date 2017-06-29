#!/usr/bin/env bash
rm -rf build
mkdir -p build

# source modules_kesch.env
module purge
module load CMake
module load craype-haswell
module load craype-network-infiniband

module use /apps/common/UES/RHAT6/easybuild/modules/all
module use /apps/escha/UES/RH6.7/sandbox-scorep/modules/all
module load Score-P/3.1-gmvapich2-15.11_cuda_7.0_gdr
module list -t
echo

#export BOOST_ROOT="/apps/escha/UES/RH6.7/easybuild/software/Boost/1.49.0-gmvolf-15.11-Python-2.7.10"
export CXX=$(which g++)
export CC=$(which gcc)
pushd build &>/dev/null
    cmake .. \
        -DCMAKE_CXX_FLAGS="-std=c++11" \
        -DENABLE_TIMER=OFF \
        -DENABLE_MPI_TIMER=OFF \
        -DENABLE_SCOREP=ON \
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

make -j 1 \
VERBOSE=1


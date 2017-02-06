#!/usr/bin/env bash
rm -rf build
mkdir -p build

module purge
source /project/c01/install/kesch/dycore/release_double/modules_cpp.env
module load CMake
module list -t
echo

export CC=gcc
export CXX=g++
export BOOST_ROOT=/apps/escha/UES/RH6.7/easybuild/software/Boost/1.49.0-gmvolf-15.11-Python-2.7.10/

pushd build &>/dev/null
    cmake .. -DBOOST_ROOT="${BOOST_ROOT}" -DCUDA_COMPUTE_CAPABILITY="sm_37" 

    make VERBOSE=1


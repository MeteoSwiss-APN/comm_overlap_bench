#!/usr/bin/env bash
source modules_daint.sh
module list 
[ -z "${jobs}" ] && jobs=4
echo "Running with ${jobs} jobs"

partition=debug
if [ "${jobs}" -gt 4 ]; then
    partition=normal
fi

export MPICH_RDMA_ENABLED_CUDA=1
#partition=debug
#export CUDA_AUTOBOOST=1
#export GCLOCK=875

export BOOST_LIBRARY_ROOT=/apps/daint/UES/jenkins/6.0.UP02/gpu/easybuild/software/Boost/1.63.0-CrayGNU-2016.11-Python-3.5.2/lib
export LD_LIBRARY_PATH=$BOOST_LIBRARY_ROOT:$LD_LIBRARY_PATH


srun --gres=gpu:1 -N $jobs -n $jobs --ntasks-per-node=1 --constraint=gpu --time=00:10:00 --partition=$partition build/src/comm_overlap_benchmark


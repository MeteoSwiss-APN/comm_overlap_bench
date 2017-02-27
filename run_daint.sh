#!/usr/bin/env bash
source modules_daint.sh
[ -z "${jobs}" ] && jobs=4
echo "Running with ${jobs} jobs"
export MPICH_RDMA_ENABLED_CUDA=1
#partition=debug
srun --gres=gpu:1 -n $jobs --ntasks-per-node=1 --cpus-per-task=1 --constraint=gpu --time=00:10:00  build/src/StandaloneStencilsCUDA


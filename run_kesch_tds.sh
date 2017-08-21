#!/usr/bin/env bash

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


echo Modules:
module list -t

# Add the boost library path manually
export BOOST_LIBRARY_PATH=/apps/escha/UES/jenkins/RH7.3-17.02/easybuild/software/boost/1.63.0-gmvolf-17.02-python-2.7.13/lib
export LD_LIBRARY_PATH=$BOOST_LIBRARY_PATH:$LD_LIBRARY_PATH

[ -z "${jobs}" ] && jobs=2
echo "Running with ${jobs} jobs (set jobs environment variable to change)"

# Setup Node configuration
tasks_socket=$jobs
if [ "$jobs" -gt "8" ]; then
    tasks_socket=8
fi
partition=debug
tasks_node=$jobs
if [ "$jobs" -gt "16" ]; then
    tasks_node=16
    partition=dev
fi
nodes=1
if [ "$jobs" -gt "16" ]; then
    let nodes=($jobs+16-1)/16
fi

#export G2G=2
## Setup GPU
[ -z "${G2G}" ] && export G2G=2
#if [ "${G2G}" == 2 ]; then
#    echo "Setting special settings for G2G=2"
#    export MV2_USE_GPUDIRECT=1
#    export MV2_CUDA_IPC=1
#    export MV2_ENABLE_AFFINITY=1
#    export MV2_GPUDIRECT_GDRCOPY_LIB=/apps/escha/gdrcopy/20170131/libgdrapi.so
#    export MV2_USE_CUDA=1
#    echo
#fi

export LD_PRELOAD=/opt/mvapich2/gdr/no-mcast/2.2/cuda8.0/mpirun/gnu4.8.5/lib64/libmpi.so

echo "Nodes: ${nodes}"
echo "Tasks/Node: ${tasks_node}"
echo "Tasks/Socket: ${tasks_socket}"
echo "Partition: ${partition}"

echo =======================================================================
echo = Default Benchmark
echo =======================================================================
srun --nodes=$nodes --ntasks=$jobs --ntasks-per-node=$tasks_node --ntasks-per-socket=$tasks_socket --partition=$partition --gres=gpu:$tasks_node --distribution=block:block --cpu_bind=q   build_tds/src/comm_overlap_benchmark
echo =======================================================================
echo = No Communication
echo =======================================================================
srun --nodes=$nodes --ntasks=$jobs --ntasks-per-node=$tasks_node --ntasks-per-socket=$tasks_socket --partition=$partition --gres=gpu:$tasks_node --distribution=block:block --cpu_bind=q  build_tds/src/comm_overlap_benchmark --nocomm
echo =======================================================================
echo = No Computation
echo =======================================================================
srun --nodes=$nodes --ntasks=$jobs --ntasks-per-node=$tasks_node --ntasks-per-socket=$tasks_socket --partition=$partition --gres=gpu:$tasks_node --distribution=block:block --cpu_bind=q  build_tds/src/comm_overlap_benchmark --nocomp


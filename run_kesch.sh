#!/usr/bin/env bash
module purge
source modules_kesch.env
echo Modules:
module list

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

# Setup GPU
[ -z "${G2G}" ] && export G2G=2
if [ "${G2G}" == 2 ]; then
    echo "Setting special settings for G2G=2"
    export MV2_USE_GPUDIRECT=1
    export MV2_CUDA_IPC=1
    export MV2_ENABLE_AFFINITY=1
    export MV2_GPUDIRECT_GDRCOPY_LIB=/apps/escha/gdrcopy/20170131/libgdrapi.so
    export MV2_USE_CUDA=1
    echo
fi


echo "Nodes: ${nodes}"
echo "Tasks/Node: ${tasks_node}"
echo "Tasks/Socket: ${tasks_socket}"
echo "Partition: ${partition}"

echo =======================================================================
echo = Default Benchmark
echo =======================================================================
srun --nodes=$nodes --ntasks=$jobs --ntasks-per-node=$tasks_node --ntasks-per-socket=$tasks_socket --partition=$partition --gres=gpu:$tasks_node --distribution=block:block --cpu_bind=q  build/src/comm_overlap_benchmark
echo =======================================================================
echo = No Communication
echo =======================================================================
srun --nodes=$nodes --ntasks=$jobs --ntasks-per-node=$tasks_node --ntasks-per-socket=$tasks_socket --partition=$partition --gres=gpu:$tasks_node --distribution=block:block --cpu_bind=q  build/src/comm_overlap_benchmark --nocomm
echo =======================================================================
echo = No Computation
echo =======================================================================
srun --nodes=$nodes --ntasks=$jobs --ntasks-per-node=$tasks_node --ntasks-per-socket=$tasks_socket --partition=$partition --gres=gpu:$tasks_node --distribution=block:block --cpu_bind=q  build/src/comm_overlap_benchmark --nocomp


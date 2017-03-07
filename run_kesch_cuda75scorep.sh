#!/usr/bin/env bash
module purge
source modules_kesch_cuda75.env
module load Score-P/3.0-gmvapich2-17.02_cuda_7.5_gdr
echo Modules:
module list

# Add the boost library path manually
export BOOST_LIBRARY_PATH=/apps/escha/UES/RH6.7/easybuild/software/Boost/1.63.0-gmvapich2-17.02_cuda_7.5_gdr-Python-2.7.12/lib
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

export SCOREP_TOTAL_MEMORY=1G
export SCOREP_ENABLE_TRACING=true
export SCOREP_CUDA_ENABLE=1

echo "Nodes: ${nodes}"
echo "Tasks/Node: ${tasks_node}"
echo "Tasks/Socket: ${tasks_socket}"
echo "Partition: ${partition}"

#srun --nodes=$nodes --ntasks=$jobs --ntasks-per-node=$tasks_node --ntasks-per-socket=$tasks_socket --partition=$partition --gres=gpu:$tasks_socket build/src/StandaloneStencilsCUDA
srun --nodes=$nodes --ntasks=$jobs --ntasks-per-node=$tasks_node --ntasks-per-socket=$tasks_socket --partition=$partition --gres=gpu:$tasks_node --distribution=block:block --cpu_bind=q  build_cuda75/src/StandaloneStencilsCUDA

#srun --nodes=$nodes --ntasks-per-node=$tasks_node --gres=gpu:$tasks_node -n $jobs -p debug  build/src/StandaloneStencilsCUDA


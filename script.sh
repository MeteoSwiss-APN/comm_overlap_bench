#!/bin/bash

module purge
module load PrgEnv-gnu

let lrank=$SLURM_LOCALID%8

GPUS=(0 1 2 3 4 5 6 7)
CPUS=(0 1 2 3 8 9 10 11)

export CUDA_VISIBLE_DEVICES=${GPUS[lrank]}

export UCX_MEMTYPE_CACHE=n
export UCX_TLS=rc_x,ud_x,mm,shm,cuda_copy,cuda_ipc,cma



#export UCX_RNDV_SCHEME=put_zcopy
#export UCX_RNDV_THRESH=2048

#export OMPI_MCA_pml=ucx
#export OMPI_MCA_btl="^vader,tcp,openib,smcuda"
#export UCX_TLS=all
#export UCX_MEMTYPE_CACHE=n

echo $SLURM_PROCID : $SLURM_LOCALID, CPU ${CPU_REORDER[$lrank]}, GPU $CUDA_VISIBLE_DEVICES, NET_DEVICE $UCX_NET_DEVICES

nvidia-smi -L
#numactl --physcpubind=${CPUS[$lrank]} ./affinity
#numactl --physcpubind=${CPUS[$lrank]} build/src/comm_overlap_benchmark


echo "CALLINB numactl --physcpubind=${CPUS[$lrank]} build/src/comm_overlap_benchmark --nocomm"
#numactl --physcpubind=${CPUS[$lrank]} 
size=512
ksize=80

echo "NOCOMM *********************************"
#build/src/comm_overlap_benchmark --nocomm --ie ${size} --je ${size} --sync --inorder
echo "NOCOMP *********************************"
#numactl --physcpubind=20 
build/src/comm_overlap_benchmark --nocomp --ie ${size} --je ${size} --ke ${ksize} --sync --inorder

#numactl --physcpubind=${CPUS[$lrank]} build/src/comm_overlap_benchmark --nocomp


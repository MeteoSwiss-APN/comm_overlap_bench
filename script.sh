#!/bin/bash

module purge
module use /apps/manali/UES/jenkins/MCH-PE20.08-UP01/NVHPC/22.3/easybuild/modules/all
module load OpenMPI

let lrank=$SLURM_LOCALID%8

GPUS=(0 1 2 3)
NICS=(mlx5_3 mlx5_2 mlx5_1 mlx5_0)
CPUS=(48 32 16 0)

export UCX_NET_DEVICES="${NICS[lrank]}:1"
export CUDA_VISIBLE_DEVICES=${GPUS[lrank]}

export UCX_RNDV_SCHEME=put_zcopy
export UCX_RNDV_THRESH=2048

export OMPI_MCA_pml=ucx
export OMPI_MCA_btl="^vader,tcp,openib,smcuda"
export UCX_TLS=all
export UCX_MEMTYPE_CACHE=n

echo $SLURM_PROCID : $SLURM_LOCALID, CPU ${CPU_REORDER[$lrank]}, GPU $CUDA_VISIBLE_DEVICES, NET_DEVICE $UCX_NET_DEVICES
size=256
ksize=80

nvidia-smi -L
numactl --physcpubind=${CPUS[$lrank]} ./affinity
echo "DEFAULT"
numactl --physcpubind=${CPUS[$lrank]} build/src/comm_overlap_benchmark --ie ${size} --je ${size} --ke ${ksize} --sync --inorder

echo "NOCOMM"
numactl --physcpubind=${CPUS[$lrank]} build/src/comm_overlap_benchmark --nocomm --ie ${size} --je ${size} --ke ${ksize} --sync --inorder

echo "NOCOMP"
numactl --physcpubind=${CPUS[$lrank]} build/src/comm_overlap_benchmark --nocomp --ie ${size} --je ${size} --ke ${ksize} --sync --inorder


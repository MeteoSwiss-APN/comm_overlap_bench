#!/bin/bash
#SBATCH --job-name=job-name
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --output=out.log
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:16
#SBATCH --partition=dev
#SBATCH --account=s83

########################################################

set -x
#We have to source the appropiate module_xxx.env before launching this submit script
module list -l

ldd $BINNAME

export GCLOCK=875
export CUDA_AUTO_BOOST=0
export OMP_NUM_THREADS=1
export MALLOC_MMAP_MAX_=0
export MALLOC_TRIM_THRESHOLD_=536870912
export G2G=1
export MV2_USE_GPUDIRECT=0 #with this perf are ok

srun -n 32 ../install/bin/StandaloneStencilsCUDA 128 128 80 --nostella --nogcl --sync

########################################################

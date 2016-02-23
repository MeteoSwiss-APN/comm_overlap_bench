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
source ../modules_cpp.env
module load Boost/1.49.0-gmvolf-15.11-Python-2.7.10
module list -l

ldd $BINNAME

export GCLOCK=875
export CUDA_AUTO_BOOST=0
export OMP_NUM_THREADS=1
export MALLOC_MMAP_MAX_=0
export MALLOC_TRIM_THRESHOLD_=536870912
export G2G=1
export MV2_USE_GPUDIRECT=0 #with this perf are ok
export MV2_IBA_EAGER_THRESHOLD=0
export MV2_SMP_EAGER_THRESHOLD=0


srun -n 32 bash ./wrap.sh 

########################################################

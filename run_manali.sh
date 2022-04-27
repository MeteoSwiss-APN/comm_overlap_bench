#!/usr/bin/env bash
#SBATCH --job-name=testsuite
#SBATCH --ntasks=9
#SBATCH --ntasks-per-node=9
#SBATCH --ntasks-per-socket=8
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --output=job.out

nodes=2
ntasks=8
ntasks_node=4
partition=normal

# module purge
# module use /apps/manali/UES/jenkins/MCH-PE20.08-UP01/NVHPC/22.3/easybuild/modules/all
# module load OpenMPI

# echo "PPP" $SLURM_LOCALID
# env
# let lrank=$SLURM_LOCALID%8

# GPUS=(0 1 2 3)
# NICS=(mlx5_3 mlx5_2 mlx5_1 mlx5_0)
# CPUS=(48 32 16 0)

# export UCX_NET_DEVICES=${NICS[lrank]}
# export CUDA_VISIBLE_DEVICES=${GPUS[lrank]}

# export UCX_RNDV_SCHEME=put_zcopy
# export UCX_RNDV_THRESH=2048

# export OMPI_MCA_pml=ucx
# export OMPI_MCA_btl="^vader,tcp,openib,smcuda"
# export UCX_TLS=all
# export UCX_MEMTYPE_CACHE=n

# echo $SLURM_PROCID : $SLURM_LOCALID, CPU ${CPU_REORDER[$lrank]}, GPU $CUDA_VISIBLE_DEVICES, $UCX_NET_DEVICES

#numactl --cpunodebind=${CPUS[$lrank]} --membind=${CPUS[$lrank]} $@


srun --partition=$partition --nodes=$nodes --ntasks=$ntasks --ntasks-per-node=$ntasks_node ./script.sh

# echo =======================================================================
# echo = Default Benchmark
# echo =======================================================================
# #srun --nodes=$nodes --ntasks=$jobs --ntasks-per-node=$tasks_node --ntasks-per-socket=$tasks_socket --partition=$partition --gres=gpu:$tasks_node --distribution=block:block --cpu_bind=q  build/src/comm_overlap_benchmark
# echo =======================================================================
# echo = No Communication
# echo =======================================================================
# #srun --nodes=$nodes --ntasks=$jobs --ntasks-per-node=$tasks_node --ntasks-per-socket=$tasks_socket --partition=$partition --gres=gpu:$tasks_node --distribution=block:block --cpu_bind=q  build/src/comm_overlap_benchmark --nocomm
# echo =======================================================================
# echo = No Computation
# echo =======================================================================
# #srun --nodes=$nodes --ntasks=$jobs --ntasks-per-node=$tasks_node --ntasks-per-socket=$tasks_socket --partition=$partition --gres=gpu:$tasks_node --distribution=block:block --cpu_bind=q  build/src/comm_overlap_benchmark --nocomp


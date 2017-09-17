jobs=$1

for argmt in " " --nocomm --nocomp ; do

G2G=1 MV2_USE_RDMA_FAST_PATH=0 \
LD_PRELOAD=`pkg-config --variable=libdir mvapich2-gdr`/libmpi.so \
srun --partition=normal -w keschcn-0001 --gres=gpu:16 -n$jobs \
--distribution=block:block --cpu_bind=q \
./GNU5.4.0_MVAPICH2-GDR2.2_CUDAV8.0.61.keschcn-0001 &> o_runjg.$jobs.$argmt

done

RRR=/apps/common/UES/sandbox/jgp/comm_overlap_bench.gitjg
export CC=$RRR/cc.scorep
export CXX=$RRR/mpicxx.scorep
export NVCC=$RRR/nvcc.scorep
export SCOREP_WRAPPER=OFF 
# export SCOREP_WRAPPER=on 

cmake \
-DMPI_VENDOR=mvapich2 \
-DENABLE_BOOST_TIMER=OFF \
-DENABLE_MPI_TIMER=OFF \
-DENABLE_SCOREP_TIMER=ON \
-DCUDA_COMPUTE_CAPABILITY="sm_37" \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_CXX_COMPILER="$CXX" \
-DCMAKE_C_COMPILER="$CC" \
-DCUDA_NVCC_EXECUTABLE="$NVCC" \
-DCUDA_HOST_COMPILER=`which g++` \
-DCMAKE_EXE_LINKER_FLAGS="-lpthread " \
$RRR

# -DCMAKE_CXX_FLAGS="-std=c++11" \
#jg -DSHARED_LIBRARY=ON \
#jg -DCMAKE_EXE_LINKER_FLAGS="-L$SLURM_DIR/lib64" \


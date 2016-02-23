#!/bin/sh

echo nvprof_file$SLURM_PROCID; 

 ../install/bin/StandaloneStencilsCUDA 128 128 80 --nostella --nogcl --sync -n 1000 

#../install/bin/StandaloneStencilsCUDA 128 128 80 --nostella --nogcl --sync -n 1000 -nh 2 --inorder
# nvprof --profile-from-start on -o nvprof_file$SLURM_PROCID ../install/bin/StandaloneStencilsCUDA 128 128 80 --nostella --nogcl --sync -n 20


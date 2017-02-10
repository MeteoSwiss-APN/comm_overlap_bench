#!/usr/bin/env bash
module purge
source modules_kesch.env
[ -z "${jobs}" ] && jobs=2
echo "Running with ${jobs} jobs"
[ -z "${G2G}" ] && export G2G=2
srun --gres=gpu:$jobs -n $jobs -p debug  build/src/StandaloneStencilsCUDA


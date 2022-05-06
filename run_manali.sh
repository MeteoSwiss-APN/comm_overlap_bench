#!/usr/bin/env bash

nodes=6
ntasks=24
ntasks_node=4
partition=normal

srun --partition=$partition --nodes=$nodes --ntasks=$ntasks --ntasks-per-node=$ntasks_node ./script.sh


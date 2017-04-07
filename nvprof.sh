#!/usr/bin/env bash
if [ -z "$EXEC" ]; then 
    echo "\$EXEC is empty. aborting"
    exit 1
fi
if [ -z "$OUTPUT_DIR" ]; then
    echo "\$OUTPUT_DIR is empty. aborting"
    exit 1
fi
stdout="${OUTPUT_DIR}/nvprof_stdout_${SLURM_PROCID}"
stderr="${OUTPUT_DIR}/nvprof_stderr_${SLURM_PROCID}"

nvprof --events all $EXEC > "${stdout}_events" 2> "${stderr}_events"
nvprof --metrics all $EXEC > "${stdout}_metrics" 2> "${stderr}_metrics"

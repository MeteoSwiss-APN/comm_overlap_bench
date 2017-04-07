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

unset COMPUTE_PROFILE
export PMI_NO_FORK=1

nvprof --metrics all     $EXEC > "${OUTPUT_DIR}/nvprof_metrics_${SLURM_PROCID}_stdout"   2> "${OUTPUT_DIR}/nvprof_metrics_${SLURM_PROCID}"
nvprof --print-gpu-trace $EXEC > "${OUTPUT_DIR}/nvprof_gpu_trace_${SLURM_PROCID}_stdout" 2> "${OUTPUT_DIR}/nvprof_gpu_trace_${SLURM_PROCID}"
nvprof --events all      $EXEC > "${OUTPUT_DIR}/nvprof_events_${SLURM_PROCID}_stdout"    2> "${OUTPUT_DIR}/nvprof_events_${SLURM_PROCID}"

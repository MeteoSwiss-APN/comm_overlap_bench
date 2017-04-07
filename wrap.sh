#!/usr/bin/env bash
if [ -z "$EXEC" ]; then 
    echo "\$EXEC is empty. aborting"
    exit 1
fi
if [ -z "$OUTPUT_DIR" ]; then
    echo "\$OUTPUT_DIR is empty. aborting"
    exit 1
fi
stdout="${OUTPUT_DIR}/stdout_${SLURM_PROCID}"
stderr="${OUTPUT_DIR}/stderr_${SLURM_PROCID}"
$EXEC > "${stdout}" 2> "${stderr}"

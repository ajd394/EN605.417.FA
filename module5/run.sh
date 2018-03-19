#!/usr/bin/env bash

set -e

FILENAME=assignment5_constant_mem

nvcc $FILENAME.cu -lcudart -Wno-deprecated-gpu-targets -o $FILENAME 

if [[ ! -v CI ]]; then
    ./$FILENAME -t 4096 -b 32
	./$FILENAME -t 8192 -b 64
    ./$FILENAME -t 8192 -b 32
fi

FILENAME=assignment5_shared_mem

nvcc $FILENAME.cu -lcudart -Wno-deprecated-gpu-targets -o $FILENAME 

if [[ ! -v CI ]]; then
    ./$FILENAME -t 32000000 -b 32
	./$FILENAME -t 32000000 -b 64
    ./$FILENAME -t 8192 -b 32
fi
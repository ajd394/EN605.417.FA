#!/usr/bin/env bash

set -e

FILENAME=assignment7

nvcc $FILENAME.cu -lcudart -Wno-deprecated-gpu-targets -o $FILENAME 

if [[ ! -v CI ]]; then
    ./$FILENAME -t 4096 -b 32
	./$FILENAME -t 32000000 -b 64
    ./$FILENAME -t 8192 -b 32
fi
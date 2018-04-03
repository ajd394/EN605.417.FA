#!/usr/bin/env bash

set -e

FILENAME=assignment8

nvcc $FILENAME.cu -lcudart -Wno-deprecated-gpu-targets -o $FILENAME 

if [[ ! -v CI ]]; then
    ./$FILENAME
fi
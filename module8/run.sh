#!/usr/bin/env bash

set -e

FILENAME=assignment8

nvcc $FILENAME.cu -lcudart -lcuda -lcublas -lcurand -I common/inc -Wno-deprecated-gpu-targets -o $FILENAME 

if [[ ! -v CI ]]; then
    ./$FILENAME
fi
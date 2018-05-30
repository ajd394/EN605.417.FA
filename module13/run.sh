#!/usr/bin/env bash

set -e

FILENAME=assignment13

if [[ -v VOC_WORKAREA ]]; then
    # Running in Vocareaum
    g++ $FILENAME.cpp -lOpenCL -I /usr/local/cuda-8.0/targets/x86_64-linux/include -Wno-unused-value -o $FILENAME 
else
    # Running on my mac 
    g++ $FILENAME.cpp -framework OpenCL -Wno-unused-value -o $FILENAME 
fi

if [[ ! -v CI ]]; then
    ./$FILENAME
fi
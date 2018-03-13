#!/usr/bin/env bash

set -e

FILENAME=assignment6

nvcc $FILENAME.cu -lcudart -o $FILENAME

if [[ ! -v CI ]]; then
    ./$FILENAME -t 1024 -b 16
    ./$FILENAME -t 256 -b 16
fi
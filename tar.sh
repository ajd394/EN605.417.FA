#!/usr/bin/env bash

set -e

declare -a files=("assignment$1" "assignment$1.cu" "run.sh" "output.txt");

pushd module"$1"
    tar -czvf ../submissions/"assignment$1".tar.gz "${files[@]}"
popd
#!/usr/bin/env bash
set -e 

modules=(
    "module3"
    "module4"
    "module5"
    "module6"
    "module7"
    "module8"
)

if [[ -v CI ]]; then
    for module in $modules; do
        echo "Building $module"
        pushd $module
            ./run.sh
        popd
    done
else
    echo "Exiting. This script is designed for CI only"
    exit 1
fi
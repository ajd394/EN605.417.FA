#!/usr/bin/env bash

set -e

FILENAME=assignment9
echo "NPP"
CURR_DIR=${FILENAME}_npp
pushd ${CURR_DIR}
    make

    if [[ ! -v CI ]]; then
        ./${CURR_DIR} --masksz 5
        ./${CURR_DIR} --masksz 10

        ./${CURR_DIR} --input ../../common/data/flower.ppm --masksz 5
        ./${CURR_DIR} --input ../../common/data/flower.ppm --masksz 10
    fi
popd

echo "NVGRAPH"
CURR_DIR=${FILENAME}_nvgraph
pushd ${CURR_DIR}
    make

    if [[ ! -v CI ]]; then
        echo "Executing twice with random edge weights"
        ./${CURR_DIR}
        ./${CURR_DIR}
    fi
popd
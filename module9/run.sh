#!/usr/bin/env bash

set -e

FILENAME=assignment9

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

CURR_DIR=${FILENAME}_nvgraph
pushd ${CURR_DIR}
    make

    if [[ ! -v CI ]]; then
        ./${CURR_DIR}
    fi
popd
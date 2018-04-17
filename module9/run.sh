#!/usr/bin/env bash

set -e

FILENAME=assignment9

printf "\nNPP\n"
CURR_DIR=${FILENAME}_npp
pushd ${CURR_DIR}
    make

    if [[ ! -v CI ]]; then
        ./${CURR_DIR} --masksz 5
        ./${CURR_DIR} --masksz 10

        #./${CURR_DIR} --input flower.ppm --masksz 5
        #./${CURR_DIR} --input flower.ppm --masksz 10
    fi
popd

printf "\nNVGRAPH\n"
CURR_DIR=${FILENAME}_nvgraph
pushd ${CURR_DIR}
    make

    if [[ ! -v CI ]]; then
        echo "Executing twice with random edge weights"
        ./${CURR_DIR}
        ./${CURR_DIR}
    fi
popd

printf "\nThrust\n"
CURR_DIR=${FILENAME}_thrust
pushd ${CURR_DIR}
    make

    if [[ ! -v CI ]]; then
        echo "Executing with varied number of elements"
        ./${CURR_DIR} -num 100
        ./${CURR_DIR} -num 1000
        ./${CURR_DIR} -num 10000
    fi
popd
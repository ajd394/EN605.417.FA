#!/usr/bin/env bash

set -e

FILENAME=assignment9

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
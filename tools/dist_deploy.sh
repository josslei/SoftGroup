#!/usr/bin/env bash
CONFIG=$1
CHECK_POINT=$2
GPUS=$3
DIST=${@:4}
PORT=${PORT:-29501}

# the last parameter
if [ "${@:$#}" == "debug" ]; then
RUN_TYPE="-m pdb"
DIST="${@:4:`expr $# - 4`}"
fi

#OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --master_port=$PORT -m pdb $(dirname "$0")/deploy.py $CONFIG $CHECK_POINT --dist ${@:4}
OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --master_port=$PORT $RUN_TYPE $(dirname "$0")/deploy.py $CONFIG $CHECK_POINT --dist $DIST

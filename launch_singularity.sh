#!/bin/bash

MODELS_SOURCE_DIR=<path to models directory> # host
MODELS_TARGET_DIR=/models

CONTAINER_NAME=ntt-llm-tools-20240216.sif

SRC_CODE_DIR=$(cd $(dirname $0);pwd)/llm-foundry/ntt
TGT_CODE_DIR=/llm-foundry/ntt/


singularity exec --nv \
	    -B ${SOURCE_DIR}:${TARGET_DIR} \
	    -B ${MODELS_SOURCE_DIR}:${MODELS_TARGET_DIR} \
	    -B ./data:/data \
	    -B ./experiments:/experiments \
	    -B ./finetune_scripts:/finetune_scripts \
	    -B ./.cache:/.cache \
	    ${CONTAINER_NAME} ${SHELL}

#docker run --gpus all \
#       -v ${SOURCE_DIR}:${TARGET_DIR} \
#       -itd --privileged=true \
#       --shm-size=32gb --ulimit memlock=-1 --ulimit stack=67108864 \
#       --name ${CONTAINER_NAME} ${IMAGE_NAME} 

#docker start ${CONTAINER_NAME}
#docker exec -it ${CONTAINER_NAME} ${SHELL}



#!/bin/sh
#$-l rt_AF=1
#$-l h_rt=24:00:00
#$-j y

# Singularity
source /etc/profile.d/modules.sh
module load singularitypro

MODELS_SOURCE_DIR=/home/acc12541ix/link/groups/work/tsuzumi_PJ/tsuzumi
MODELS_TARGET_DIR=/models

SINGULARITY_ROOT_DIR=/home/acc12541ix/link/groups/work/tsuzumi_PJ/tsuzumi_dialogue
#IMAGE_NAME=ntt-llm_tools:20240216
CONTAINER_NAME=$SINGULARITY_ROOT_DIR/ntt-llm-tools-20240216.sif

SRC_CODE_DIR=$SINGULARITY_ROOT_DIR/llm-foundry/ntt
TGT_CODE_DIR=/llm-foundry/ntt/

prog='/finetune_scripts/run/run_multi_full_abci.sh'
#prog='/finetune_scripts/run/run_multi_lora_abci.sh'

singularity exec --nv \
	    -B ${SOURCE_DIR}:${TARGET_DIR} \
	    -B ${MODELS_SOURCE_DIR}:${MODELS_TARGET_DIR} \
	    -B $SINGULARITY_ROOT_DIR/data:/data \
	    -B $SINGULARITY_ROOT_DIR/experiments:/experiments \
	    -B $SINGULARITY_ROOT_DIR/finetune_scripts:/finetune_scripts \
	    -B $SINGULARITY_ROOT_DIR/.cache:/.cache \
	    ${CONTAINER_NAME} ${SHELL} \
	    $prog	    

#docker run --gpus all \
#       -v ${SOURCE_DIR}:${TARGET_DIR} \
#       -itd --privileged=true \
#       --shm-size=32gb --ulimit memlock=-1 --ulimit stack=67108864 \
#       --name ${CONTAINER_NAME} ${IMAGE_NAME} 

#docker start ${CONTAINER_NAME}
#docker exec -it ${CONTAINER_NAME} ${SHELL}



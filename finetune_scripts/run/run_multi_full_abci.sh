#!/bin/bash
version='v1.5'
#dataname='favdial_context6_persona5_4'
dataname='all_context6_persona5r_step'

export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="tsuzumi_dialogue"
export WANDB_ENTITY="cs-dialog"
export WANDB_NAME=$version-$dataname-$JOB_ID
#echo $WANDB_NAME

prog='/finetune_scripts/src/lora.py'


# INPUT-DATA PATH
training_data='/data/converted_data/'$version'_training_data/'$dataname'/train.jsonl'
num_train_data=$(wc -l < $training_data)
valid_data='/data/converted_data/'$version'_training_data/'$dataname'/valid.jsonl'
cache_dir='/.cache/'

# INPUT-LLM PATH
base_modelname='/models/v1_02-7b-instruct-hf/'

# RESUME PATH
#resume_dirname=<resume model dirname>

# OUTPUT-MODEL PATH
save_dirname='/experiments/output/'$version'_training_data/'$dataname

# TRAIN PARAM
epochs=3 #データ数が少ない時は長めに学習してからよいcheckpointを探す
per_device_batch_size=1 #GPUに乗る範囲で大きくするのがよい
total_batch_size=32 #実質的なバッチサイズ．経験的にこのくらい
#lr=3e-4 #5e-4だと高すぎる印象．データ数と相談しながら調整
lr=1e-5
max_seq_len=1024 # 必要に応じて2048
eval_steps=300
save_steps=600
n_worker=2
#dtype=fp32
dtype=bf16
save_total_limit=5

export PATH=/composer-python/:$PATH
#python $prog \
#python3.10 -m torch.distributed.launch \

accelerate launch --config_file /finetune_scripts/run/configs/accelerate_config_zero3.yaml \
	$prog \
	--train_data $training_data \
	--num_train_data $num_train_data \
	--eval_data $valid_data \
	--cache_dir $cache_dir \
	--per_device_batch_size $per_device_batch_size \
	--total_batch_size $total_batch_size \
	--do_full_finetune \
	--dataloader_num_workers $n_worker \
	--epochs $epochs \
	--target_modules $target_modules \
	--base_modelname $base_modelname \
	--evaluation_steps $eval_steps \
	--save_dirname $save_dirname \
	--save_steps $save_steps \
        --lr $lr \
	--max_seq_len $max_seq_len \
        --model_dtype $dtype

#--resume_dirname $resume_dirname


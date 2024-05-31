import sys, os
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
from transformers.trainer import set_seed

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from datasets import load_dataset
from IPython import embed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import Accelerator

#def main(local_rank, world_size, args):
def main(args):
    set_seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(args.base_modelname,
                                              use_fast=False,
                                              trust_remote_code=True)
    tokenizer.padding_side = "right"
    collator = DataCollatorForSeq2Seq(tokenizer,
            padding='max_length',
            max_length=args.max_seq_len)

    def make_and_tokenized_format(example):
        tokenized_text = tokenizer(
            example["prompt"],
            text_target =example["response"] +tokenizer.eos_token,
            truncation=False,
            return_token_type_ids=False,
        )
        input_ids =tokenized_text["input_ids"] + tokenized_text["labels"]
        labels =[-100] * len(tokenized_text["input_ids"]) + tokenized_text["labels"]
        if len(input_ids) > args.max_seq_len:
            input_ids =input_ids[:args.max_seq_len]
            labels =labels[:args.max_seq_len]
        else:
            #DataCollatorForSeq2Seqはlabelsをpadしてくれない
            labels =labels + [-100] * (args.max_seq_len - len(labels))

        return {"input_ids": input_ids, "labels": labels}

    # Data setup
    train_filename = args.train_data
    train_dataset = load_dataset('json', data_files=train_filename, cache_dir =args.cache_dir, split='train')
    tokenized_dataset = train_dataset.map(make_and_tokenized_format)

    eval_filename = args.eval_data
    eval_dataset = load_dataset('json', data_files=eval_filename, cache_dir=args.cache_dir, split='train')
    eval_tokenized_dataset = eval_dataset.map(make_and_tokenized_format)

    # Train Param
    learning_rate = args.lr
    warmup_steps = args.warmup_steps
    logging_steps = args.logging_steps
    save_steps = args.save_steps
    epochs = args.epochs

    # Model setup
    device_map = "auto"

    batch_size = args.total_batch_size
    gradient_accumulation_steps = batch_size // (args.per_device_batch_size)
    
    #ddpをonにするとOOMが起きる．onにした場合の挙動は未検証なのでdatalodaerやgasの設定に注意
    ddp = False #world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    elif world_size > 1:
        #accelerateからの起動
        device_map ={"": Accelerator().process_index}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    #------------------------------------------------        
    # Set model_dtype
    if args.model_dtype is not None:
        model_dtype = get_dtype(args.model_dtype)
    else:
        model_dtype = torch.float32

    fp16_flag = False
    bf16_flag = False
    if args.model_dtype == "fp16":
        fp16_flag = True
    elif args.model_dtype == "bf16":
        bf16_flag = True

    
    config = AutoConfig.from_pretrained(args.base_modelname,
                                        trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_modelname,
                                                 config=config,
                                                 #load_in_8bit=True,
                                                 torch_dtype=model_dtype,
                                                 #device_map=device_map,
                                                 trust_remote_code=True)
    #model = prepare_model_for_int8_training(model,
    #                                        use_gradient_checkpointing=False)

    if not args.do_full_finetune:
        # LoRA setup
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=0.05,
            bias="none",
            fan_in_fan_out=False,
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)

    # Resume
    resume_from_checkpoint = args.resume_dirname
    """
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")
    """
            
    if not args.do_full_finetune:            
        model.print_trainable_parameters()
    
    # training
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1  gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
        #args.per_device_batch_size *=torch.cuda.device_count()
    if ddp:
        model =model.to(local_rank)
        model =DDP(model, device_ids =[local_rank])

    save_model_dir = args.save_dirname
    max_steps = int(args.num_train_data/batch_size * epochs)
    local_rank = args.local_rank
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        #num_train_epochs=1,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=fp16_flag,
        bf16=bf16_flag,
        logging_steps=logging_steps,
        logging_dir = "./tensorboard_logs",
        optim="adamw_torch",
        evaluation_strategy="steps",
        eval_steps=args.evaluation_steps,
        output_dir=save_model_dir,
        save_strategy="steps",
        save_steps=save_steps,
        #save_only_model = True,
        #load_best_model_at_end=True,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=True,
        #report_to="tensorboard",
        report_to="wandb",
        local_rank=local_rank,
        lr_scheduler_type =args.lr_scheduler_type,
        dataloader_num_workers =args.dataloader_num_workers
    )

    if world_size ==1 or Accelerator().process_index ==0:
        print(training_args)

    trainer = Trainer(
        model=model,
        data_collator=collator,
        args=training_args,
        train_dataset=tokenized_dataset.with_format("torch"),
        eval_dataset=eval_tokenized_dataset.with_format("torch"),
    )

    if ddp:
        dist.barrier()

    # train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # save
    model.save_pretrained(save_model_dir)

    
def get_dtype(dtype: str):
    if dtype == 'fp32':
        return torch.float32
    elif dtype == 'fp16':
        return torch.float16
    elif dtype == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError(
            f'dtype {dtype} is not supported. ' +\
            f'We only support fp32, fp16, and bf16 currently')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tuning NTT model.',
                                     add_help=True)
    parser.add_argument('--train_data', required=True, type=str)
    parser.add_argument('--cache_dir', default=None, type=str)
    parser.add_argument('--num_train_data', required=True, type=int)
    parser.add_argument('--seed', required=False, type=int, default =42)
    parser.add_argument('--per_device_batch_size', required=True, type=int)
    parser.add_argument('--total_batch_size', required=True, type=int)
    parser.add_argument('--do_full_finetune', required=False, default=False, action='store_true')
    parser.add_argument('--lora_r', required=False, type=int)
    parser.add_argument('--lora_alpha', required=False, type=int)
    parser.add_argument('--base_modelname', required=False, type=str)
    parser.add_argument('--save_dirname', required=True, type=str)
    parser.add_argument('--resume_dirname', required=False, type=str,
                        default=None)
    # 20240213 
    #   指定した数をnとして、訓練データを1/nにする（n個ごとにサンプリングする）引数。現時点では利用されていないため、一旦コメントアウトとする。
    # parser.add_argument('--skip_sample_num', required=False, type=int,
    #                     default=0)
    parser.add_argument('--local-rank', required=False, type=int,
                        default=0)
    parser.add_argument('--target_modules', required=False, type=str, nargs ="*",
                        default=["Wqkv"])
    # Add args
    parser.add_argument('--lr', required=False, type=float,
                        default=3e-4)
    parser.add_argument('--warmup_steps', required=False, type=int,
                        default=0)
    parser.add_argument('--logging_steps', required=False, type=int,
                        default=20)
    parser.add_argument('--save_steps', required=False, type=int,
                        default=100) # Only save the best model
    parser.add_argument('--lr_scheduler_type', required=False, type=str,
                        default='cosine')
    parser.add_argument('--max_seq_len', required=False, type=int,
                        default=2048)
    parser.add_argument('--epochs', required=False, type=int,
                        default=1)
    parser.add_argument('--dataloader_num_workers', required=False, type=int,
                        default=0)
    parser.add_argument('--model_dtype', required=False, type=str,
                        default='fp16', choices=['bf16','fp16','fp32'])
    parser.add_argument('--save_total_limit', required=False, type=int,
                        default=10)

    # Add evaluation args
    parser.add_argument('--eval_data', required=True, type=str)
    parser.add_argument('--evaluation_steps', required=False, type=int,
                        default=100)
    
    args = parser.parse_args()
    main(args)

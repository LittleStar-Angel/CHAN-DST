#!/bin/bash
set -xe

data=multiwoz2.1
model=model_adaptive_finetune
batch=32
lr=1e-5
seed=1111
pretrain=exp/multiwoz2.1-update/model_lamb0.5_batch32_lr1e-4_seed1111_mtdrop0.1/loss.best
output_postfix=${model}_batch${batch}_lr${lr}_seed${seed}
target_slot='all'
bert_dir='/data/shanyong/.pytorch_pretrained_bert'

# train
python3 code/main_adaptive_finetune.py --do_train --num_train_epochs 20 --data_dir data/${data} --bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --model ${model} --output_dir exp/${data}/${output_postfix} --target_slot $target_slot --warmup_proportion -1 --learning_rate ${lr} --train_batch_size ${batch} --gradient_accumulation_steps 2 --eval_batch_size 16 --distance_metric euclidean --patience 3 --tf_dir tensorboard/${data}/${output_postfix} --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 --seed ${seed} --focal --gamma 2 --reload ${pretrain}

CUDA_VISIBLE_DEVICES=0 python3 code/main_adaptive_finetune.py --do_eval --num_train_epochs 300 --data_dir data/${data} --bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --model ${model} --output_dir exp/${data}/${output_postfix} --target_slot $target_slot --warmup_proportion 0.1 --learning_rate ${lr} --train_batch_size ${batch} --gradient_accumulation_steps 1 --eval_batch_size 8 --distance_metric euclidean --patience 15 --tf_dir tensorboard/${data}/${output_postfix} --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 

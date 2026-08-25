#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p ./logs/LongForecasting/WF151_24

export CUDA_VISIBLE_DEVICES=0

model_name=Autoformer

seq_lens=(144)
bss=(1024)

# muti
pred_lens=(1 2 4 8 16)
lrs=(1e-3)
dropouts=(0.2)
d_models=(256)
n_layers=(3)

# solo
# pred_lens=(16)
# lrs=(5e-4)
# dropouts=(0.0)
# d_models=(512)
# n_layers=(1)

#
#
# 检查是否为 for n_layer in "${n_layers[@]}"; do……………………………………………………………�?n_layer/n_layers
#
#
# enc_in 输入变量�?# 日志目录
log_dir="./logs/LongForecasting/WF151_24/"
for pred_len in "${pred_lens[@]}"; do
  for seq_len in "${seq_lens[@]}"; do
    for bs in "${bss[@]}"; do
      for lr in "${lrs[@]}"; do
        for dropout in "${dropouts[@]}"; do
          for d_model in "${d_models[@]}"; do
            for n_layer in "${n_layers[@]}"; do
                          cmd="python -u run.py \
                            --task_name long_term_forecast_Transformer \
                            --is_training 1 \
                            --root_path dataset/WF151/ \
                            --data_path WF151-24.csv \
                            --model_id ${model_name}_WF151_24_${seq_len}_${pred_len} \
                            --model $model_name \
                            --data Special \
                            --features MS \
                            --seq_len $seq_len \
                            --pred_len $pred_len \
                            --batch_size $bs \
                            --learning_rate $lr \
                            --enc_in 151 \
                            --dec_in 151 \
                            --c_out 151 \
                            --pe_type no\
                            --freq t\
                            --train_epochs 100\
                            --n_layers $n_layer \
                            --dropout $dropout \
                            --d_model $d_model \
                            --no_amp \
                            --des 'Exp' \
                            --itr 1 >${log_dir}${model_name}_WF151_24_${seq_len}_${pred_len}_${dropout}_${d_model}_bz${bs}_lr${lr}_nl${n_layer}.log"

                          eval $cmd
            done
          done
        done
      done
    done
  done
done
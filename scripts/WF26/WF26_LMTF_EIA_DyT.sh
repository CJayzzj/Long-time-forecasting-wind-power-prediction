#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p ./logs/LongForecasting/WF26

export CUDA_VISIBLE_DEVICES=0

model_name=LMTF_EIA_DyT

seq_lens=(144) 
bss=(1024)

# muti: 测试多个预测长度
pred_lens=(1 2 4 8 16)
lrs=(1e-3)  
dropouts=(0.2)  
d_models=(256)
n_layers=(3)     

# solo: 单个参数用于快速测试
# pred_lens=(16)
# lrs=(5e-4)
# dropouts=(0.0)
# d_models=(512) 
# n_layers=(1)

# 日志目录
log_dir="./logs/LongForecasting/WF26/"
for pred_len in "${pred_lens[@]}"; do
  for seq_len in "${seq_lens[@]}"; do
    for bs in "${bss[@]}"; do
      for lr in "${lrs[@]}"; do
        for dropout in "${dropouts[@]}"; do
          for d_model in "${d_models[@]}"; do
            for n_layer in "${n_layers[@]}"; do
                          cmd="python -u run.py \
                            --task_name long_term_forecast \
                            --is_training 1 \
                            --root_path dataset/WF26/ \
                            --data_path WF26.csv \
                            --model_id ${model_name}_WF26_${seq_len}_${pred_len} \
                            --model $model_name \
                            --data Special \
                            --features MS \
                            --seq_len $seq_len \
                            --pred_len $pred_len \
                            --batch_size $bs \
                            --learning_rate $lr \
                            --enc_in 26 \
                            --dec_in 1 \
                            --c_out 1 \
                            --pe_type no\
                            --freq t\
                            --train_epochs 100\
                            --n_layers $n_layer \
                            --dropout $dropout \
                            --d_model $d_model \
                            --des 'Exp' \
                            --itr 1 >${log_dir}${model_name}_WF26_${seq_len}_${pred_len}_${dropout}_${d_model}_bz${bs}_lr${lr}_nl${n_layer}.log"

                          eval $cmd
            done
          done
        done
      done
    done
  done
done

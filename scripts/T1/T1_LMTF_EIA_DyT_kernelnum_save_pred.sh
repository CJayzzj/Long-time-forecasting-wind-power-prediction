#!/bin/bash
#  卷积核个数实验 (num_kernels)：测试不同数量的多尺度卷积核
# - seq_len=144, d_model=256, n_layers=3, dropout=0.2, batch_size=1024, lr=0.001
# - 测试所有pred_len: 1, 2, 4, 8, 16
cd "$(dirname "$0")/../.."
echo "工作目录: $(pwd)"
mkdir -p ./logs/LongForecasting/T1/kernelnums_DyT_pred/
log_dir="./logs/LongForecasting/T1/kernelnums_DyT_pred/"

export CUDA_VISIBLE_DEVICES=0

model_name=LMTF_EIA_DyT_kernel

seq_len=144
pred_lens=(1 2 4 8 16)
enc_in=4
dec_in=1
d_model=256
n_layers=3
n_heads=8
dropout=0.2
batch_size=1024
learning_rate=0.001
train_epochs=100
patience=6
num_workers=4

declare -A kernel_configs
kernel_configs["K1"]="25"
kernel_configs["K2"]="25_13"
kernel_configs["K3"]="25_13_7"
kernel_configs["K4"]="25_13_7_5"
kernel_configs["K5"]="25_13_7_5_3"

for config_name in K1 K2 K3 K4 K5; do
    kernel_sizes=${kernel_configs[$config_name]}
    for pred_len in "${pred_lens[@]}"; do
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path dataset/T1/ \
          --data_path T1.csv \
          --model_id ${model_name}_T1_${config_name}_sl${seq_len}_pl${pred_len} \
          --model $model_name \
          --data Special \
          --features MS \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --enc_in $enc_in \
          --dec_in $dec_in \
          --d_model $d_model \
          --n_layers $n_layers \
          --n_heads $n_heads \
          --dropout $dropout \
          --batch_size $batch_size \
          --learning_rate $learning_rate \
          --train_epochs $train_epochs \
          --patience $patience \
          --num_workers $num_workers \
          --pe_type no \
          --freq t \
          --des 'Exp' \
          --itr 1 \
          --kernel_sizes "$kernel_sizes" \
          >${log_dir}${model_name}_${config_name}_pl${pred_len}.log
    done
done


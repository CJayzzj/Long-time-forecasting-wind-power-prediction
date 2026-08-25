#!/bin/bash
# ============================================================
# LMTF-EIA (woResAttn版本) 候选模型验证实验
#
# 背景: 消融实验表明 woResAttn (忘记保留EID，用MLP替代注意力块) 效果更优，
#       将 woResAttn（不用MLP，保留EID，然后直接投影残差） 
#
# 模型: LMTF-EIA (woResAttn) - 残差支路: EID → 直接投影 (无注意力块)
#
#   seq_len=144, d_model=256, n_layers=3, dropout=0.2
#   batch_size=1024, lr=1e-3, features=MS
#   enc_in=4, dec_in=1, c_out=1
#
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p ./logs/LongForecasting/T1

export CUDA_VISIBLE_DEVICES=0

model_name=LMTF_EIA_ablation
ablation_type=woResAttn

# 固定参数
seq_len=144
bs=1024
lr=1e-3
dropout=0.2
d_model=256
n_layer=3

# 变化参数 - 仅预测长度
pred_lens=(1 2 4 8 16)

# 日志目录
log_dir="./logs/LongForecasting/T1/"

echo "=========================================="
echo "Running LMTF-EIA woResAttn (SOTA variant)"
echo "=========================================="

for pred_len in "${pred_lens[@]}"; do
  echo "Training with pred_len=$pred_len ..."
  
  cmd="python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path dataset/T1/ \
    --data_path T1.csv \
    --model_id LMTF_EIA_woResAttn_T1_${seq_len}_${pred_len} \
    --model $model_name \
    --ablation_type $ablation_type \
    --data Special \
    --features MS \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --batch_size $bs \
    --learning_rate $lr \
    --enc_in 4 \
    --dec_in 1 \
    --c_out 1 \
    --pe_type no \
    --freq t \
    --train_epochs 100 \
    --n_layers $n_layer \
    --dropout $dropout \
    --d_model $d_model \
    --des 'Exp' \
    --itr 1 >${log_dir}LMTF_EIA_woResAttn_T1_${seq_len}_${pred_len}.log"
  
  eval $cmd
  
  echo "Completed pred_len=$pred_len. Log: ${log_dir}LMTF_EIA_woResAttn_T1_${seq_len}_${pred_len}.log"
done

echo "=========================================="
echo "All woResAttn experiments completed!"
echo "=========================================="

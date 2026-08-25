#!/bin/bash
# ============================================================
# LMTF-EIA + Dynamic Tanh 消融实验 (Ablation Study)
#
# 目标:
#   在 Dynamic Tanh 替代 LayerNorm 的前提下，继续验证 EIA 机制各组件贡献。
#
# 消融设置:
#   0. full      - 完整 DyT 模型
#   1. woEIA     - 去掉整个 EIA 机制
#   2. woEID     - 去掉事件强度检测，改用自注意力
#   3. woResAttn - 去掉注意力块，保留 EID 直连 Linear_res
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p ./logs/LongForecasting/T1/dyt_ablation

export CUDA_VISIBLE_DEVICES=0

model_name=LMTF_EIA_DyT_ablation
seq_len=144
d_model=256
n_layers=3
n_heads=8
dropout=0.2
lr=1e-3
bs=1024
train_epochs=100
patience=6
num_workers=4

pred_lens=(1 2 4 8 16)
log_dir="./logs/LongForecasting/T1/dyt_ablation/"

run_ablation() {
    local ablation_type=$1
    local pred_len=$2

    echo "  Running ${ablation_type}, pred_len=${pred_len}..."

    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path dataset/T1/ \
        --data_path T1.csv \
        --model_id LMTF_EIA_DyT_ablation_${ablation_type}_pl${pred_len} \
        --model $model_name \
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
        --train_epochs $train_epochs \
        --patience $patience \
        --n_layers $n_layers \
        --n_heads $n_heads \
        --dropout $dropout \
        --d_model $d_model \
        --num_workers $num_workers \
        --ablation_type $ablation_type \
        --des 'DyT_Ablation' \
        --itr 1 \
        >${log_dir}LMTF-EIA-DyT_${ablation_type}_pl${pred_len}.log 2>&1
}

echo "=============================================="
echo "LMTF-EIA + DyT 消融实验开始"
echo "时间: $(date)"
echo "=============================================="

echo ">>> [0/3] Full Baseline (LMTF-EIA_DyT)"
for pred_len in "${pred_lens[@]}"; do
    run_ablation "full" $pred_len
done
echo "    Full Baseline 完成"

echo ">>> [1/3] 模块级消融: w/o EIA"
for pred_len in "${pred_lens[@]}"; do
    run_ablation "woEIA" $pred_len
done
echo "    w/o EIA 完成"

echo ">>> [2/3] 组件级消融: w/o EID"
for pred_len in "${pred_lens[@]}"; do
    run_ablation "woEID" $pred_len
done
echo "    w/o EID 完成"

echo ">>> [3/3] 组件级消融: w/o ResAttn"
for pred_len in "${pred_lens[@]}"; do
    run_ablation "woResAttn" $pred_len
done
echo "    w/o ResAttn 完成"

echo "=============================================="
echo "LMTF-EIA + DyT 消融实验全部完成"
echo "时间: $(date)"
echo "总计: 4组 × 5步长 = 20个实验"
echo "结果: test_dict/T1/records.txt"
echo "日志: ${log_dir}"
echo "=============================================="
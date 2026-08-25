#!/bin/bash
# ============================================================
# LMTF-EIA 模型消融实验 (Ablation Study)
# 
# 聚焦Innovation 1 (EIA机制) 的层次化消融:
#   模块级消融 (Module-level):
#     1. woEIA     - 去掉整个EIA机制           → 验证Innovation 1整体
#   
#   组件级消融 (Component-level, Innovation 1细化):
#     2. woEID     - 自注意力替代事件驱动交叉注意力 → 验证EID组件
#     3. woResAttn - MLP替代注意力块            → 验证注意力机制
#
# 参数与SOTA实验 (T1_LMTF_EIA.sh) 完全一致:
#   seq_len=144, d_model=256, n_layers=3, dropout=0.2
#   batch_size=1024, lr=1e-3, features=MS
#   enc_in=4, dec_in=1, c_out=1
#
# 总实验量: 4组(含full baseline) × 5步长 = 20个实验
# ============================================================

cd /home/apulis-dev/code/20251209_ZZJ

mkdir -p ./logs/LongForecasting/T1/ablation

export CUDA_VISIBLE_DEVICES=0 

# ==================== 统一参数配置 ====================
model_name=LMTF_EIA_ablation
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
log_dir="./logs/LongForecasting/T1/ablation/"

# ==================== 公共参数模板 ====================
run_ablation() {
    local ablation_type=$1
    local pred_len=$2
    
    echo "  Running ${ablation_type}, pred_len=${pred_len}..."
    
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path dataset/T1/ \
        --data_path T1.csv \
        --model_id LMTF_EIA_ablation_${ablation_type}_pl${pred_len} \
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
        --des 'Ablation' \
        --itr 1 \
        >${log_dir}LMTF-EIA_${ablation_type}_pl${pred_len}.log 2>&1
}

echo "=============================================="
echo "LMTF-EIA 消融实验开始"
echo "时间: $(date)"
echo "=============================================="

# ============================================================
# 0. Full Baseline (完整模型)
# ============================================================
echo ">>> [0/3] Full Baseline (LMTF-EIA)"
for pred_len in "${pred_lens[@]}"; do
    run_ablation "full" $pred_len
done
echo "    Full Baseline 完成"

# ============================================================
# 1. 模块级消融: w/o EIA (整个事件强度注意力机制)
#    去掉EventIntensityDetector和ResidualAttentionBlock
#    残差分量不经过任何注意力处理，直接投影到预测长度
# ============================================================
echo ">>> [1/3] 模块级消融: w/o EIA"
for pred_len in "${pred_lens[@]}"; do
    run_ablation "woEIA" $pred_len
done
echo "    w/o EIA 完成"

# ============================================================
# 2. 组件级消融: w/o EID (Event Intensity Detector)
#    保留注意力机制，但用自注意力(Q=K=V=residual)替代
#    事件强度驱动的交叉注意力(Q=residual, K=V=event_intensity)
# ============================================================
echo ">>> [2/3] 组件级消融: w/o EID"
for pred_len in "${pred_lens[@]}"; do
    run_ablation "woEID" $pred_len
done
echo "    w/o EID 完成"

# ============================================================
# 3. 组件级消融: w/o ResAttn (Residual Attention)
#    用2层MLP替代注意力块处理残差分量
# ============================================================
echo ">>> [3/3] 组件级消融: w/o ResAttn"
for pred_len in "${pred_lens[@]}"; do
    run_ablation "woResAttn" $pred_len
done
echo "    w/o ResAttn 完成"

echo "=============================================="
echo "LMTF-EIA 消融实验全部完成"
echo "时间: $(date)"
echo "总计: 4组 × 5步长 = 20个实验"
echo "结果: test_dict/T1/records.txt"
echo "日志: ${log_dir}"
echo "=============================================="

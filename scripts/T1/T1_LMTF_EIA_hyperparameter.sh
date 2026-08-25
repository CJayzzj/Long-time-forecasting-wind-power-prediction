#!/bin/bash
# ============================================================
# LMTF-EIA模型超参数敏感性实验
# 验证模型对d_model, n_layers的敏感程度
# ============================================================

# 切换到项目根目录 (脚本所在目录的上两级)
cd "$(dirname "$0")/../.."
echo "工作目录: $(pwd)"

mkdir -p ./logs/LongForecasting/T1/hyperparameter

export CUDA_VISIBLE_DEVICES=0

model_name=LMTF_EIA
seq_len=144
pred_len=16  # 待定 选择一个代表性的预测步长

# 基准配置
base_d_model=256
base_n_layers=3
base_n_heads=8
base_dropout=0.2
base_lr=1e-3
base_bs=1024

log_dir="./logs/LongForecasting/T1/hyperparameter/"

echo "=============================================="
echo "超参数敏感性实验开始"
echo "=============================================="

# ============================================================
# 实验1: d_model敏感性实验
# 固定n_layers=3, n_heads=8, 变化d_model
# ============================================================
echo ">>> 实验1: d_model敏感性实验"
d_models=(64 128 256 512)

for d_model in "${d_models[@]}"; do
    echo "Running d_model=$d_model..."
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path dataset/T1/ \
        --data_path T1.csv \
        --model_id ${model_name}_T1_hp_dmodel_${d_model} \
        --model $model_name \
        --data Special \
        --features MS \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --batch_size $base_bs \
        --learning_rate $base_lr \
        --enc_in 4 \
        --dec_in 4 \
        --c_out 4 \
        --pe_type no \
        --freq t \
        --train_epochs 100 \
        --n_layers $base_n_layers \
        --n_heads $base_n_heads \
        --dropout $base_dropout \
        --d_model $d_model \
        --des 'HP_dmodel' \
        --itr 1 >${log_dir}${model_name}_hp_dmodel_${d_model}_pl${pred_len}.log
done

# ============================================================
# 实验2: n_layers敏感性实验
# 固定d_model=256, n_heads=8, 变化n_layers
# ============================================================
echo ">>> 实验2: n_layers敏感性实验"
n_layers_list=(1 2 3 4 5)

for n_layer in "${n_layers_list[@]}"; do
    echo "Running n_layers=$n_layer..."
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path dataset/T1/ \
        --data_path T1.csv \
        --model_id ${model_name}_T1_hp_nlayers_${n_layer} \
        --model $model_name \
        --data Special \
        --features MS \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --batch_size $base_bs \
        --learning_rate $base_lr \
        --enc_in 4 \
        --dec_in 4 \
        --c_out 4 \
        --pe_type no \
        --freq t \
        --train_epochs 100 \
        --n_layers $n_layer \
        --n_heads $base_n_heads \
        --dropout $base_dropout \
        --d_model $base_d_model \
        --des 'HP_nlayers' \
        --itr 1 >${log_dir}${model_name}_hp_nlayers_${n_layer}_pl${pred_len}.log
done

echo "=============================================="
echo "超参数敏感性实验完成"
echo "=============================================="
echo ""

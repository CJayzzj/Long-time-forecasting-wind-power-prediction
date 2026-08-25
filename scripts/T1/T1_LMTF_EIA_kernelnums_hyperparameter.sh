#!/bin/bash

# =============================================================================
# LMTF-EIA模型 MultiScaleMovingAverage 模块的 kernel_sizes 超参数实验
# 
# 实验设计：
# 1. 卷积核个数实验 (num_kernels)：测试不同数量的多尺度卷积核
# 2. 卷积核大小实验 (kernel_scale)：测试不同的卷积核大小组合
# 
# 固定参数+控制变量：保持与原始LMTF-EIA SOTA实验一致
# - seq_len=144, d_model=256, n_layers=3, dropout=0.2, batch_size=1024, lr=0.001
# - 测试所有pred_len: 1, 2, 4, 8, 16
# =============================================================================

# 切换到项目根目录 (脚本所在目录的上两级)
cd "$(dirname "$0")/../.."
echo "工作目录: $(pwd)"

# 创建日志目录
mkdir -p ./logs/LongForecasting/T1/kernelnums_hyperparameter
log_dir="./logs/LongForecasting/T1/kernelnums_hyperparameter/"

export CUDA_VISIBLE_DEVICES=0

model_name=LMTF_EIA_kernel

# 公共参数
seq_len=144  # 与原始实验一致
pred_lens=(1 2 4 8 16)  # 测试所有预测长度
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
num_workers=4  # 数据加载并行进程数，提高GPU利用率

# =============================================================================
# 实验1: 卷积核个数实验 (Number of Kernels)
# 测试不同数量的卷积核对模型性能的影响
# 从1个卷积核（单尺度基准）到6个卷积核（多尺度）
# =============================================================================
echo "=============================================="
echo "实验1: 卷积核个数实验 (Number of Kernels)"
echo "=============================================="

# 定义不同的kernel配置
declare -A kernel_configs
kernel_configs["num1"]="25"
kernel_configs["num2"]="25_13"
kernel_configs["num3"]="25_13_7"
kernel_configs["num4"]="25_13_7_5"
kernel_configs["num5"]="25_13_7_5_3"
kernel_configs["num6"]="25_13_7_5_3_1"

# 遍历每个kernel配置
for config_name in num1 num2 num3 num4 num5 num6; do
    kernel_sizes=${kernel_configs[$config_name]}
    echo ""
    echo ">>> Running: kernel_${config_name} [${kernel_sizes//_/, }]"
    
    # 遍历所有pred_len
    for pred_len in "${pred_lens[@]}"; do
        echo "    pred_len=${pred_len}..."
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path dataset/T1/ \
          --data_path T1.csv \
          --model_id ${model_name}_T1_kernel_${config_name}_${seq_len}_pl${pred_len} \
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
          --kernel_sizes "$kernel_sizes" \
          >${log_dir}LMTF_EIA_kernel_${config_name}_pl${pred_len}.log
    done
done

echo ""
echo "=============================================="
echo "卷积核个数实验完成!"
echo "=============================================="
echo ""
echo "实验总结:"
echo "  控制变量: seq_len=144, d_model=256, n_layers=3, dropout=0.2"
echo "  测试pred_len: 1, 2, 4, 8, 16"
echo ""
echo "  实验1 - 卷积核个数 (验证多尺度vs单尺度):"
echo "    - 1个: [25] (单尺度基准)"
echo "    - 2个: [25, 13]"
echo "    - 3个: [25, 13, 7]"
echo "    - 4个: [25, 13, 7, 5]"
echo "    - 5个: [25, 13, 7, 5, 3] (默认)"
echo "    - 6个: [25, 13, 7, 5, 3, 1]"
echo ""
echo "  总实验数: 6种配置 × 5种pred_len = 30个实验"

#!/bin/bash

# =============================================================================
# LMTF-EIA模型 MultiScaleMovingAverage 模块的 kernel_scale 超参数实验
# 
# 实验设计：
# 基于kernelnum实验已验证多尺度优于单尺度，固定5个卷积核，测试不同的卷积核大小组合
# 
# 固定参数+控制变量：保持与原始LMTF-EIA SOTA实验一致
# - seq_len=144, d_model=256, n_layers=3, dropout=0.2, batch_size=1024, lr=0.001
# - 测试所有pred_len: 1, 2, 4, 8, 16
# =============================================================================

# 切换到项目根目录 (脚本所在目录的上两级)
cd "$(dirname "$0")/../.."
echo "工作目录: $(pwd)"

# 创建日志目录
mkdir -p ./logs/LongForecasting/T1/kernelscale_hyperparameter
log_dir="./logs/LongForecasting/T1/kernelscale_hyperparameter/"

export CUDA_VISIBLE_DEVICES=0

model_name=LMTF_EIA_kernel

# 公共参数 (与T1_LMTF_EIA.sh保持一致)
seq_len=144
pred_lens=(1 2 4 8 16)
enc_in=4
dec_in=1
c_out=1
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
# 实验: 卷积核大小实验 (Kernel Scale)
# 测试不同尺度的卷积核组合对模型性能的影响
# 固定5个卷积核，改变卷积核的大小范围
# =============================================================================
echo "=============================================="
echo "实验: 卷积核大小实验 (Kernel Scale)"
echo "=============================================="

# 定义不同的kernel scale配置
declare -A kernel_configs
kernel_configs["xlarge"]="49_37_25_13_7"      # 超大尺度: 更大的感受野
kernel_configs["large"]="37_25_17_9_5"        # 大尺度: 偏大的卷积核
kernel_configs["default"]="25_13_7_5_3"       # 默认尺度: 原始配置
kernel_configs["small"]="17_13_9_5_3"         # 小尺度: 偏小的卷积核
kernel_configs["xsmall"]="13_9_7_5_3"         # 超小尺度: 更小的感受野
kernel_configs["uniform"]="21_17_13_9_5"      # 均匀间隔: 等差数列式
kernel_configs["power2"]="33_17_9_5_3"        # 近2的幂次: 指数递减风格

# 遍历每个kernel配置
for config_name in xlarge large default small xsmall uniform power2; do
    kernel_sizes=${kernel_configs[$config_name]}
    # 将下划线分隔转换为逗号分隔用于显示
    kernel_display=${kernel_sizes//_/, }
    echo ""
    echo ">>> Running: kernel_${config_name} [${kernel_display}]"
    
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
          --c_out $c_out \
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
          >${log_dir}LMTF_EIA_kernel_${config_name}_pl${pred_len}.log
    done
done

echo ""
echo "=============================================="
echo "卷积核大小实验完成!"
echo "=============================================="
echo ""
echo "实验总结:"
echo "  控制变量: seq_len=144, d_model=256, n_layers=3, dropout=0.2"
echo "  测试pred_len: 1, 2, 4, 8, 16"
echo ""
echo "  卷积核大小配置 (固定5个卷积核):"
echo "    - xlarge:  [49, 37, 25, 13, 7]  (超大尺度)"
echo "    - large:   [37, 25, 17, 9, 5]   (大尺度)"
echo "    - default: [25, 13, 7, 5, 3]    (默认尺度)"
echo "    - small:   [17, 13, 9, 5, 3]    (小尺度)"
echo "    - xsmall:  [13, 9, 7, 5, 3]     (超小尺度)"
echo "    - uniform: [21, 17, 13, 9, 5]   (均匀间隔)"
echo "    - power2:  [33, 17, 9, 5, 3]    (近2的幂次)"
echo ""
echo "  总实验数: 7种配置 × 5种pred_len = 35个实验"

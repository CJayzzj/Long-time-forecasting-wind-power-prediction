#!/bin/bash
# Multi-seed stability experiment for LMTF_EIA_DyT
# sl=144, pl in {1,2,4,8,16}, 12 random seeds
# Results are appended to records.txt via the existing exp pipeline
# After completion, run: python visual/plot_stability_violin.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p ./logs/LongForecasting/T1/multiseed

export CUDA_VISIBLE_DEVICES=0

model_name=LMTF_EIA_DyT
seq_len=144
pred_lens=(1 2 4 8 16)
seeds=(2025 42 43 44 45 46 47 48 49 50 51 52)

log_dir="./logs/LongForecasting/T1/multiseed/"

total=$(( ${#pred_lens[@]} * ${#seeds[@]} ))
count=0

for pred_len in "${pred_lens[@]}"; do
  for seed in "${seeds[@]}"; do
    count=$(( count + 1 ))
    echo "=========================================="
    echo " [$count/$total] pl=${pred_len}  seed=${seed}"
    echo "=========================================="

    python -u run.py \
      --seed $seed \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path dataset/T1/ \
      --data_path T1.csv \
      --model_id ${model_name}_T1_${seq_len}_${pred_len} \
      --model $model_name \
      --data Special \
      --features MS \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --batch_size 1024 \
      --learning_rate 1e-3 \
      --enc_in 4 \
      --dec_in 1 \
      --c_out 1 \
      --pe_type no \
      --freq t \
      --train_epochs 100 \
      --n_layers 3 \
      --dropout 0.2 \
      --d_model 256 \
      --des Exp \
      --itr 1 \
      > "${log_dir}${model_name}_T1_${seq_len}_${pred_len}_seed${seed}.log" 2>&1

    echo "Done: pl=${pred_len} seed=${seed}  ->  ${log_dir}${model_name}_T1_${seq_len}_${pred_len}_seed${seed}.log"
  done
done

echo ""
echo "All $total experiments completed."
echo "Now run:  python visual/plot_stability_violin.py"

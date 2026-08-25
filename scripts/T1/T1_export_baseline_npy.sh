#!/bin/bash
# Export npy files for baseline models (test-only, no training)
# Requires existing checkpoints from previous training runs.
# After completion, run: python visual/plot_curve_scatter_4models.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p ./logs/LongForecasting/T1/export_npy

export CUDA_VISIBLE_DEVICES=0

pred_lens=(1 2 4 8 16)
seq_len=144

# ── Leddam (task: long_term_forecast) ──
for pl in "${pred_lens[@]}"; do
  echo "=== Leddam  sl=${seq_len} pl=${pl} ==="
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --root_path dataset/T1/ \
    --data_path T1.csv \
    --model_id Leddam_T1_${seq_len}_${pl} \
    --model Leddam \
    --data Special \
    --features MS \
    --seq_len $seq_len \
    --pred_len $pl \
    --batch_size 1024 \
    --learning_rate 1e-3 \
    --enc_in 4 \
    --dec_in 1 \
    --c_out 1 \
    --pe_type no \
    --freq t \
    --n_layers 3 \
    --dropout 0.2 \
    --d_model 256 \
    --des Exp \
    --itr 1 \
    > "./logs/LongForecasting/T1/export_npy/Leddam_pl${pl}.log" 2>&1
done

# ── iTransformer (task: long_term_forecast_Transformer) ──
for pl in "${pred_lens[@]}"; do
  echo "=== iTransformer  sl=${seq_len} pl=${pl} ==="
  python -u run.py \
    --task_name long_term_forecast_Transformer \
    --is_training 0 \
    --root_path dataset/T1/ \
    --data_path T1.csv \
    --model_id iTransformer_T1_${seq_len}_${pl} \
    --model iTransformer \
    --data Special \
    --features MS \
    --seq_len $seq_len \
    --pred_len $pl \
    --batch_size 1024 \
    --learning_rate 1e-3 \
    --enc_in 4 \
    --dec_in 1 \
    --c_out 1 \
    --pe_type no \
    --freq t \
    --n_layers 3 \
    --dropout 0.2 \
    --d_model 256 \
    --des Exp \
    --itr 1 \
    > "./logs/LongForecasting/T1/export_npy/iTransformer_pl${pl}.log" 2>&1
done

# ── DLinear (task: long_term_forecast_Transformer) ──
for pl in "${pred_lens[@]}"; do
  echo "=== DLinear  sl=${seq_len} pl=${pl} ==="
  python -u run.py \
    --task_name long_term_forecast_Transformer \
    --is_training 0 \
    --root_path dataset/T1/ \
    --data_path T1.csv \
    --model_id DLinear_T1_${seq_len}_${pl} \
    --model DLinear \
    --data Special \
    --features MS \
    --seq_len $seq_len \
    --pred_len $pl \
    --batch_size 1024 \
    --learning_rate 1e-3 \
    --enc_in 4 \
    --dec_in 1 \
    --c_out 1 \
    --pe_type no \
    --freq t \
    --n_layers 3 \
    --dropout 0.2 \
    --d_model 256 \
    --des Exp \
    --itr 1 \
    > "./logs/LongForecasting/T1/export_npy/DLinear_pl${pl}.log" 2>&1
done

echo ""
echo "All baseline npy exports completed."
echo "Expected files in test_dict/T1/:"
echo "  Leddam_sl144_pl{1,2,4,8,16}_{pred,true}.npy"
echo "  iTransformer_sl144_pl{1,2,4,8,16}_{pred,true}.npy"
echo "  DLinear_sl144_pl{1,2,4,8,16}_{pred,true}.npy"
echo ""
echo "Now run:  python visual/plot_curve_scatter_4models.py"

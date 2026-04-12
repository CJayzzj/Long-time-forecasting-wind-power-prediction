# LMTF-EIA: Long-Term Wind Power Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)

A deep learning research repository for long-term wind power time series forecasting.  
The core model is **LMTF-EIA** (**L**earnable **M**ulti-scale **T**rend **F**usion with **E**vent **I**ntensity **A**ttention), a dual-branch architecture that combines multi-scale temporal decomposition with event-driven attention for robust wind power prediction.

---

## Table of Contents

- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Baselines](#baselines)
- [Results](#results)
- [Citation](#citation)

---

## Model Architecture

LMTF-EIA is a dual-branch time series forecasting architecture with two complementary components:

### 1. Learnable Multi-scale Trend Fusion (LMTF)
- Applies learnable decomposition (LD) to separate trend and seasonal signals at multiple scales.
- Fuses multi-scale trend representations via the Multi-Scale Moving Average (MSMA) module.
- Uses Dynamic Tanh (DyT) activations to adaptively handle non-stationary wind patterns.
- Reversible Instance Normalization (RevIN) is applied at input and output to handle distributional shift.

### 2. Event Intensity Attention (EIA)
- Captures abrupt meteorological events (e.g., wind ramp events, storms) that are poorly represented by smooth trend components.
- Event Intensity Detection (EID) blocks identify high-gradient segments in the input series.
- Attention scores are modulated by event intensity, focusing model capacity on critical forecast windows.

### Dual-Branch Fusion
The trend branch (LMTF) and event branch (EIA) outputs are fused via a learned gating mechanism before the final projection head.

```
Input ──► RevIN ──┬──► LD Decomposition ──► MSMA ──► Trend Branch ──┐
                  │                                                    ├──► Gate ──► Output
                  └──► EID Blocks ──► EIA Attention ──► Event Branch ─┘
```

---

## Project Structure

```
Long-time-forecasting-wind-power-prediction/
├── models/             # Model definitions
│   └── LMTF_EIA.py    # Core LMTF-EIA model and baseline wrappers
├── layers/             # Reusable neural network modules
│   ├── RevIN.py        # Reversible Instance Normalization
│   ├── DyT.py          # Dynamic Tanh activation
│   ├── LD.py           # Learnable Decomposition
│   ├── MSMA.py         # Multi-Scale Moving Average
│   └── EID.py          # Event Intensity Detection & Attention
├── data_provider/      # Dataset loading and preprocessing pipelines
│   ├── data_loader.py  # Dataset classes (wind power, benchmark datasets)
│   └── data_factory.py # Dataset factory and DataLoader builders
├── exp/                # Experiment runners
│   ├── exp_main.py     # Training and evaluation loop
│   └── exp_basic.py    # Base experiment class
├── scripts/            # Shell scripts for batch experiments
│   └── run_experiments.sh
├── utils/              # Helper functions
│   ├── metrics.py      # MAE, MSE, RMSE, MAPE evaluation metrics
│   ├── tools.py        # Early stopping, learning rate adjustment
│   └── timefeatures.py # Time feature engineering
├── visual/             # Visualization scripts
│   ├── plot_results.py # Forecast vs. ground truth plots
│   └── attention_maps.py # EIA attention weight visualizations
├── checkpoints/        # Saved model weights (git-ignored)
├── logs/               # Training logs (git-ignored)
├── excel/              # Results tables (git-ignored)
├── requirements.txt    # Python dependencies
└── run.py              # Main entry point
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU recommended (CUDA 11.7+ / 12.x)

### 1. Clone the Repository

```bash
git clone https://github.com/CJayzzj/Long-time-forecasting-wind-power-prediction.git
cd Long-time-forecasting-wind-power-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: Install PyTorch separately following the [official guide](https://pytorch.org/get-started/locally/) to match your CUDA version, e.g.:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Data Preparation

Place your dataset files in a `dataset/` directory at the project root (this directory is git-ignored).

```
dataset/
├── wind_power/
│   └── wind_power.csv
└── ETTh1.csv           # Optional: benchmark datasets for comparison
```

Wind power CSV files are expected to contain columns: `date`, `power`, and optional meteorological features (`wind_speed`, `wind_direction`, `temperature`, etc.).

---

## Usage

### Training

Train LMTF-EIA on a wind power dataset:

```bash
python run.py \
  --model LMTF_EIA \
  --data wind_power \
  --data_path dataset/wind_power/wind_power.csv \
  --target power \
  --seq_len 336 \
  --pred_len 96 \
  --d_model 512 \
  --n_heads 8 \
  --e_layers 2 \
  --d_ff 2048 \
  --dropout 0.1 \
  --train_epochs 100 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --patience 10 \
  --checkpoints ./checkpoints \
  --gpu 0
```

Or use the provided shell scripts:

```bash
bash scripts/run_experiments.sh
```

#### Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--model` | Model name (`LMTF_EIA`, `Transformer`, `Autoformer`, …) | `LMTF_EIA` |
| `--seq_len` | Input sequence (look-back window) length | `336` |
| `--pred_len` | Forecast horizon length | `96` |
| `--d_model` | Model dimension | `512` |
| `--n_heads` | Number of attention heads | `8` |
| `--e_layers` | Number of encoder layers | `2` |
| `--dropout` | Dropout rate | `0.1` |
| `--train_epochs` | Maximum training epochs | `100` |
| `--batch_size` | Training batch size | `32` |
| `--learning_rate` | Initial learning rate | `1e-4` |
| `--patience` | Early stopping patience | `10` |
| `--gpu` | GPU device index | `0` |

### Evaluation

Evaluate a trained checkpoint on the test set:

```bash
python run.py \
  --model LMTF_EIA \
  --data wind_power \
  --data_path dataset/wind_power/wind_power.csv \
  --target power \
  --seq_len 336 \
  --pred_len 96 \
  --checkpoints ./checkpoints \
  --gpu 0 \
  --is_training 0
```

### Visualization

Generate forecast plots and attention maps after evaluation:

```bash
python visual/plot_results.py --results_path ./results
python visual/attention_maps.py --checkpoint ./checkpoints/<run_id>/checkpoint.pth
```

---

## Baselines

The following baseline models are included for comparison:

| Model | Reference |
|---|---|
| Transformer | [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) |
| Autoformer | [Wu et al., NeurIPS 2021](https://arxiv.org/abs/2106.13008) |
| FEDformer | [Zhou et al., ICML 2022](https://arxiv.org/abs/2201.12740) |
| PatchTST | [Nie et al., ICLR 2023](https://arxiv.org/abs/2211.14730) |
| iTransformer | [Liu et al., ICLR 2024](https://arxiv.org/abs/2310.06625) |
| TimesNet | [Wu et al., ICLR 2023](https://arxiv.org/abs/2210.02186) |
| DLinear | [Zeng et al., AAAI 2023](https://arxiv.org/abs/2205.13504) |

---

## Results

Results are saved as Excel files in `excel/` (git-ignored locally).  
Below is a representative performance summary on wind power forecasting:

| Model | Horizon | MAE | MSE | RMSE |
|---|---|---|---|---|
| **LMTF-EIA** | 96 | — | — | — |
| **LMTF-EIA** | 192 | — | — | — |
| **LMTF-EIA** | 336 | — | — | — |
| **LMTF-EIA** | 720 | — | — | — |

> Detailed results will be updated upon paper publication.

---

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{lmtf_eia_2024,
  title   = {LMTF-EIA: Learnable Multi-scale Trend Fusion with Event Intensity Attention
             for Long-Term Wind Power Forecasting},
  author  = {CJayzzj},
  year    = {2024},
  url     = {https://github.com/CJayzzj/Long-time-forecasting-wind-power-prediction}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

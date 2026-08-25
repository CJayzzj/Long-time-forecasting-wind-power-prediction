import argparse
import os
import random
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import torch


def find_project_root(start_path):
    current = os.path.abspath(start_path)
    while True:
        has_exp = os.path.isdir(os.path.join(current, "exp"))
        has_run = os.path.isfile(os.path.join(current, "run.py"))
        if has_exp and has_run:
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise RuntimeError(
        "Cannot locate project root containing both 'exp/' and 'run.py'. "
        "Please run this script inside your project tree."
    )


ROOT_DIR = find_project_root(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_long_term_forecasting_Transformer import Exp_Long_Term_Forecast_Transformer


PRED_LENS = [1, 2, 4, 8, 16]


def build_setting(args):
    return (
        f"{args.data_path[:-4]}_pl{args.pred_len}_n_layers_{args.n_layers}_"
        f"d_model_{args.d_model}_dropout_{args.dropout}_pe_type_{args.pe_type}_"
        f"bs_{args.batch_size}_lr_{args.learning_rate}"
    )


def build_args(overrides):
    defaults = {
        "task_name": "long_term_forecast",
        "is_training": 0,
        "model_id": "export_visual_only",
        "model": "LMTF_EIA",
        "data": "Special",
        "root_path": "dataset/T1/",
        "data_path": "T1.csv",
        "features": "MS",
        "target": "OT",
        "freq": "t",
        "checkpoints": "./checkpoints",
        "seq_len": 144,
        "label_len": 72,
        "pred_len": 1,
        "enc_in": 4,
        "d_model": 256,
        "dec_in": 1,
        "c_out": 1,
        "n_layers": 3,
        "n_heads": 8,
        "pe_type": "no",
        "dropout": 0.2,
        "revin": True,
        "num_workers": 4,
        "itr": 1,
        "train_epochs": 1,
        "batch_size": 1024,
        "patience": 6,
        "learning_rate": 1e-3,
        "des": "ExportVisual",
        "loss": "mse",
        "lradj": "type1",
        "use_amp": True,
        "no_amp": False,
        "use_gpu": torch.cuda.is_available(),
        "gpu": 0,
        "use_multi_gpu": False,
        "devices": "0,1,2,3",
        "device_ids": [0],
        "embed": "timeF",
        "factor": 1,
        "d_ff": 512,
        "e_layers": 3,
        "d_layers": 1,
        "activation": "gelu",
        "patch_length": 16,
        "patch_stride": 8,
        "decp_scale": 1.0,
        "output_attention": False,
        "kernel_size": 25,
        "moving_avg": 25,
        "ablation_type": "full",
        "kernel_sizes": "default",
    }
    defaults.update(overrides)

    # 与 run.py 保持一致
    defaults["label_len"] = defaults["seq_len"] // 2
    if defaults["task_name"] == "long_term_forecast_Transformer":
        defaults["e_layers"] = defaults["n_layers"]

    return SimpleNamespace(**defaults)


def pick_exp_class(task_name):
    if task_name == "long_term_forecast":
        return Exp_Long_Term_Forecast
    if task_name == "long_term_forecast_Transformer":
        return Exp_Long_Term_Forecast_Transformer
    raise ValueError(f"Unsupported task_name: {task_name}")


def expected_visual_candidates(args):
    data_name = args.data_path[:-4]
    # exp_long_term_forecasting.py
    p1 = os.path.join(
        ROOT_DIR,
        "visual",
        data_name,
        (
            f"{args.model}_{data_name}_sl{args.seq_len}_pl{args.pred_len}_"
            f"n_layers_{args.n_layers}_d_model_{args.d_model}_dropout_{args.dropout}_"
            f"pe_type_{args.pe_type}_bs_{args.batch_size}_lr_{args.learning_rate}.png"
        ),
    )
    # exp_long_term_forecasting_Transformer.py
    p2 = os.path.join(
        ROOT_DIR,
        "visual",
        data_name,
        (
            f"{data_name}_{args.model}_pl{args.pred_len}_n_layers_{args.n_layers}_"
            f"d_model_{args.d_model}_dropout_{args.dropout}_pe_type_{args.pe_type}_"
            f"bs_{args.batch_size}_lr_{args.learning_rate}.png"
        ),
    )
    return [p1, p2]


def _strip_model_prefix(state_dict):
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("model.") for k in keys):
        return {k[len("model."):]: v for k, v in state_dict.items()}
    return state_dict


def _add_model_prefix(state_dict):
    return {f"model.{k}": v for k, v in state_dict.items()}


def load_checkpoint_compat(model, checkpoint_file):
    ckpt = torch.load(checkpoint_file, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

    attempts = [ckpt, _strip_model_prefix(ckpt), _add_model_prefix(ckpt)]
    errors = []
    for sd in attempts:
        try:
            model.load_state_dict(sd, strict=True)
            return
        except RuntimeError as e:
            errors.append(str(e))

    raise RuntimeError(
        "Checkpoint/model mismatch. This usually means checkpoints were overwritten by another experiment "
        "that shares the same setting directory name.\n"
        + "\n---\n".join(errors)
    )


def export_one_case(case, out_root):
    args = build_args(case["args"])
    setting = build_setting(args)
    checkpoint_file = os.path.join(ROOT_DIR, args.checkpoints, setting, "checkpoint.pth")

    if not os.path.exists(checkpoint_file):
        return False, f"MISSING_CHECKPOINT: {checkpoint_file}"

    before_files = {p: (os.path.getmtime(p) if os.path.exists(p) else None) for p in expected_visual_candidates(args)}

    ExpClass = pick_exp_class(args.task_name)
    exp = ExpClass(args)

    # 兼容不同模型封装下的checkpoint键名（如 model.xxx）
    load_checkpoint_compat(exp.model, checkpoint_file)
    exp.test(setting, test=0)

    produced = None
    for p in expected_visual_candidates(args):
        if os.path.exists(p):
            old_time = before_files[p]
            new_time = os.path.getmtime(p)
            if old_time is None or new_time >= old_time:
                produced = p
                break

    if produced is None:
        return False, "VISUAL_NOT_FOUND_AFTER_TEST"

    os.makedirs(out_root, exist_ok=True)
    out_file = os.path.join(out_root, f"{case['tag']}.png")
    shutil.copy2(produced, out_file)
    return True, out_file


def build_kernelnums_cases():
    kernel_configs = {
        "num1": "25",
        "num2": "25_13",
        "num3": "25_13_7",
        "num4": "25_13_7_5",
        "num5": "25_13_7_5_3",
        "num6": "25_13_7_5_3_1",
    }
    cases = []
    for cfg_name, kernel_sizes in kernel_configs.items():
        for pred_len in PRED_LENS:
            cases.append(
                {
                    "tag": f"kernelnums_{cfg_name}_pl{pred_len}_{kernel_sizes}",
                    "args": {
                        "task_name": "long_term_forecast",
                        "model": "LMTF_EIA_kernel",
                        "pred_len": pred_len,
                        "kernel_sizes": kernel_sizes,
                    },
                }
            )
    return cases


def build_kernelscale_cases():
    kernel_configs = {
        "xlarge": "49_37_25_13_7",
        "large": "37_25_17_9_5",
        "default": "25_13_7_5_3",
        "small": "17_13_9_5_3",
        "xsmall": "13_9_7_5_3",
        "uniform": "21_17_13_9_5",
        "power2": "33_17_9_5_3",
    }
    cases = []
    for cfg_name, kernel_sizes in kernel_configs.items():
        for pred_len in PRED_LENS:
            cases.append(
                {
                    "tag": f"kernelscale_{cfg_name}_pl{pred_len}_{kernel_sizes}",
                    "args": {
                        "task_name": "long_term_forecast",
                        "model": "LMTF_EIA_kernel",
                        "pred_len": pred_len,
                        "kernel_sizes": kernel_sizes,
                    },
                }
            )
    return cases


def build_dyt_ablation_cases():
    ablations = ["full", "woEIA", "woEID", "woResAttn"]
    cases = []
    for abl in ablations:
        for pred_len in PRED_LENS:
            cases.append(
                {
                    "tag": f"dyt_ablation_{abl}_pl{pred_len}",
                    "args": {
                        "task_name": "long_term_forecast",
                        "model": "LMTF_EIA_DyT_ablation",
                        "pred_len": pred_len,
                        "ablation_type": abl,
                    },
                }
            )
    return cases


def build_sota_cases():
    # 与 scripts/T1 下脚本保持一致
    model_specs = [
        ("Leddam", "long_term_forecast", 1, 1),
        ("LMTF_EIA", "long_term_forecast", 1, 1),
        ("LMTF_EIA_DyT", "long_term_forecast", 1, 1),
        ("DLinear", "long_term_forecast_Transformer", 1, 1),
        ("iTransformer", "long_term_forecast_Transformer", 1, 1),
        ("PatchTST", "long_term_forecast_Transformer", 1, 1),
        ("TiDE", "long_term_forecast_Transformer", 1, 1),
        ("Autoformer", "long_term_forecast_Transformer", 4, 4),
        ("FEDformer", "long_term_forecast_Transformer", 4, 4),
    ]

    cases = []
    for model, task_name, dec_in, c_out in model_specs:
        for pred_len in PRED_LENS:
            cases.append(
                {
                    "tag": f"sota_{model}_pl{pred_len}",
                    "args": {
                        "task_name": task_name,
                        "model": model,
                        "pred_len": pred_len,
                        "dec_in": dec_in,
                        "c_out": c_out,
                    },
                }
            )
    return cases


def resolve_groups(group_name):
    if group_name == "kernelnums":
        return {"kernelnums": build_kernelnums_cases()}
    if group_name == "kernelscale":
        return {"kernelscale": build_kernelscale_cases()}
    if group_name == "dyt_ablation":
        return {"dyt_ablation": build_dyt_ablation_cases()}
    if group_name == "sota":
        return {"sota": build_sota_cases()}
    if group_name == "all":
        return {
            "kernelnums": build_kernelnums_cases(),
            "kernelscale": build_kernelscale_cases(),
            "dyt_ablation": build_dyt_ablation_cases(),
            "sota": build_sota_cases(),
        }
    raise ValueError(f"Unknown group: {group_name}")


def main():
    parser = argparse.ArgumentParser(description="T1实验结果批量可视化导出（仅test，不训练）")
    parser.add_argument(
        "--group",
        type=str,
        default="all",
        choices=["kernelnums", "kernelscale", "dyt_ablation", "sota", "all"],
        help="导出实验组",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints",
        help="checkpoint 根目录（相对项目根目录）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./visual/T1/exported",
        help="导出图片目录（相对项目根目录）",
    )
    parser.add_argument("--gpu", type=int, default=0, help="单卡编号")
    parser.add_argument("--cpu", action="store_true", help="强制CPU导图")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    grouped_cases = resolve_groups(args.group)

    ok_count = 0
    fail_count = 0

    for group_name, cases in grouped_cases.items():
        print("=" * 80)
        print(f"[EXPORT GROUP] {group_name} | total={len(cases)}")
        print("=" * 80)

        group_out = os.path.join(ROOT_DIR, args.out_dir, group_name)
        os.makedirs(group_out, exist_ok=True)

        for idx, case in enumerate(cases, 1):
            case["args"]["checkpoints"] = args.checkpoints
            case["args"]["gpu"] = args.gpu
            case["args"]["use_gpu"] = (not args.cpu) and torch.cuda.is_available()

            print(f"[{idx}/{len(cases)}] {case['tag']}")
            try:
                success, info = export_one_case(case, group_out)
            except Exception as e:
                success, info = False, f"EXCEPTION: {e}"
            if success:
                ok_count += 1
                print(f"  OK   -> {info}")
            else:
                fail_count += 1
                print(f"  FAIL -> {info}")

    print("\n" + "=" * 80)
    print(f"Export finished. success={ok_count}, failed={fail_count}")
    print(f"Output root: {os.path.join(ROOT_DIR, args.out_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

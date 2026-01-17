"""
主程序 - LeNet 多优化器对比实验
运行 3 种优化器 × 3 个随机种子 = 9 次实验
"""
import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any

import torch
import torch.nn as nn

from config import get_train_config, get_optimizer_configs, TrainConfig
from utils import set_seed, get_system_info
from model import create_model, count_parameters
from dataset import get_dataloaders, get_dataset_info
from optimizers import build_optimizer, get_optimizer_info
from trainer import Trainer, RunMetrics


def run_single_experiment(
    optimizer_name: str,
    seed: int,
    cfg: TrainConfig,
    device: str,
) -> RunMetrics:
    """运行单个实验"""
    
    # 设置随机种子
    set_seed(seed)
    
    # 创建运行 ID
    run_id = f"{optimizer_name}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    
    # 创建模型
    model = create_model(num_classes=10, device=device)
    
    # 获取优化器配置并创建优化器
    opt_configs = get_optimizer_configs()
    opt_cfg = opt_configs[optimizer_name]
    optimizer = build_optimizer(optimizer_name, model.parameters(), opt_cfg, model=model)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        run_id=run_id,
        optimizer_name=optimizer_name,
        seed=seed,
        log_dir=cfg.log_dir,
        checkpoint_dir=cfg.checkpoint_dir,
        warmup_steps=cfg.warmup_steps,
        gpu_sample_interval_ms=cfg.gpu_sample_interval_ms,
    )
    
    # 训练
    metrics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=cfg.epochs,
        eval_freq=cfg.eval_freq,
    )
    
    return metrics


def run_all_experiments(cfg: TrainConfig, device: str) -> List[RunMetrics]:
    """运行所有实验"""
    
    all_metrics: List[RunMetrics] = []
    optimizer_names = ["SGD", "AdamW", "Muon"]
    
    total_runs = len(optimizer_names) * len(cfg.seeds)
    current_run = 0
    
    for optimizer_name in optimizer_names:
        for seed in cfg.seeds:
            current_run += 1
            print(f"\n{'#'*60}")
            print(f"# Run {current_run}/{total_runs}: {optimizer_name} with seed {seed}")
            print(f"{'#'*60}")
            
            try:
                metrics = run_single_experiment(
                    optimizer_name=optimizer_name,
                    seed=seed,
                    cfg=cfg,
                    device=device,
                )
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Error in {optimizer_name} with seed {seed}: {e}")
                import traceback
                traceback.print_exc()
    
    return all_metrics


def save_summary(all_metrics: List[RunMetrics], cfg: TrainConfig):
    """保存汇总结果"""
    import numpy as np
    from collections import defaultdict
    
    # 按优化器分组
    grouped = defaultdict(list)
    for m in all_metrics:
        grouped[m.optimizer].append(m)
    
    summary = {}
    for opt_name, metrics_list in grouped.items():
        best_val_accs = [m.best_val_acc for m in metrics_list]
        test_acc_bests = [m.test_acc_best for m in metrics_list]
        test_acc_lasts = [m.test_acc_last for m in metrics_list]
        avg_step_times = [m.avg_step_time_ms for m in metrics_list]
        peak_mems = [m.peak_mem_allocated_mb for m in metrics_list]
        gpu_utils = [m.gpu_util_mean for m in metrics_list]
        
        summary[opt_name] = {
            "best_val_acc_mean": float(np.mean(best_val_accs)),
            "best_val_acc_std": float(np.std(best_val_accs)),
            "test_acc_best_mean": float(np.mean(test_acc_bests)),
            "test_acc_best_std": float(np.std(test_acc_bests)),
            "test_acc_last_mean": float(np.mean(test_acc_lasts)),
            "test_acc_last_std": float(np.std(test_acc_lasts)),
            "avg_step_time_mean": float(np.mean(avg_step_times)),
            "avg_step_time_std": float(np.std(avg_step_times)),
            "peak_mem_mean": float(np.mean(peak_mems)),
            "peak_mem_std": float(np.std(peak_mems)),
            "gpu_util_mean": float(np.mean(gpu_utils)),
            "gpu_util_std": float(np.std(gpu_utils)),
            "best_val_epochs": [m.best_val_epoch for m in metrics_list],
        }
    
    # 保存汇总
    summary_path = os.path.join(cfg.result_dir, "summary.json")
    os.makedirs(cfg.result_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 打印汇总表格
    print("\n" + "="*100)
    print("FINAL SUMMARY")
    print("="*100)
    print(f"{'Optimizer':<10} | {'Best Val Acc':<18} | {'Test Acc (Best)':<18} | "
          f"{'Test Acc (Last)':<18} | {'Step Time (ms)':<15} | {'Peak Mem (MB)':<15}")
    print("-"*100)
    
    for opt_name in ["SGD", "AdamW", "Muon"]:
        if opt_name in summary:
            s = summary[opt_name]
            print(f"{opt_name:<10} | "
                  f"{s['best_val_acc_mean']:.2f} ± {s['best_val_acc_std']:.2f}      | "
                  f"{s['test_acc_best_mean']:.2f} ± {s['test_acc_best_std']:.2f}      | "
                  f"{s['test_acc_last_mean']:.2f} ± {s['test_acc_last_std']:.2f}      | "
                  f"{s['avg_step_time_mean']:.2f} ± {s['avg_step_time_std']:.2f}    | "
                  f"{s['peak_mem_mean']:.1f} ± {s['peak_mem_std']:.1f}")
    
    print("="*100)
    print(f"\nSummary saved to: {summary_path}")
    
    return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LeNet Optimizer Comparison")
    parser.add_argument("--optimizer", type=str, default=None,
                        help="Run only specified optimizer (SGD, AdamW, Muon)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run only specified seed")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    args = parser.parse_args()
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 获取配置
    cfg = get_train_config()
    
    # 覆盖配置
    if args.epochs:
        cfg.epochs = args.epochs
    if args.seed:
        cfg.seeds = [args.seed]
    
    # 打印系统信息
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"{key}: {value}")
    
    # 打印数据集信息
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    dataset_info = get_dataset_info()
    for key, value in dataset_info.items():
        print(f"{key}: {value}")
    
    # 打印优化器配置
    print("\n" + "="*60)
    print("OPTIMIZER CONFIGURATIONS")
    print("="*60)
    opt_configs = get_optimizer_configs()
    for name, opt_cfg in opt_configs.items():
        print(f"\n{name}:")
        for key, value in vars(opt_cfg).items():
            print(f"  {key}: {value}")
    
    # 保存实验配置
    os.makedirs(cfg.result_dir, exist_ok=True)
    config_path = os.path.join(cfg.result_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            "train_config": vars(cfg),
            "system_info": sys_info,
            "dataset_info": dataset_info,
            "optimizer_configs": {name: vars(c) for name, c in opt_configs.items()},
        }, f, indent=2, default=str)
    
    # 运行实验
    if args.optimizer:
        # 只运行指定优化器
        all_metrics = []
        for seed in cfg.seeds:
            metrics = run_single_experiment(
                optimizer_name=args.optimizer,
                seed=seed,
                cfg=cfg,
                device=device,
            )
            all_metrics.append(metrics)
    else:
        # 运行所有实验
        all_metrics = run_all_experiments(cfg, device)
    
    # 保存汇总
    if all_metrics:
        save_summary(all_metrics, cfg)
    
    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()

"""
可视化模块 - 生成实验结果图表和报告
"""
import os
import json
import glob
from typing import Dict, List, Any
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_all_metrics(log_dir: str = "logs") -> Dict[str, List[Dict]]:
    """加载所有实验的指标"""
    grouped = defaultdict(list)
    
    for metrics_file in glob.glob(os.path.join(log_dir, "*_metrics.json")):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            grouped[metrics['optimizer']].append(metrics)
    
    return dict(grouped)


def plot_accuracy_curves(
    grouped_metrics: Dict[str, List[Dict]],
    output_dir: str = "results"
):
    """绘制准确率曲线图"""
    os.makedirs(output_dir, exist_ok=True)
    
    colors = {'SGD': '#2ecc71', 'AdamW': '#3498db', 'Muon': '#e74c3c'}
    
    # 图1: Train/Val Accuracy vs Epoch
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for opt_name, metrics_list in grouped_metrics.items():
        all_train_accs = []
        all_val_accs = []
        epochs = None
        
        for metrics in metrics_list:
            epoch_metrics = metrics['epoch_metrics']
            epochs = [em['epoch'] for em in epoch_metrics]
            train_accs = [em['train_acc'] for em in epoch_metrics]
            val_accs = [em['val_acc'] for em in epoch_metrics]
            all_train_accs.append(train_accs)
            all_val_accs.append(val_accs)
        
        if epochs is None:
            continue
        
        # 计算均值和标准差
        train_mean = np.mean(all_train_accs, axis=0)
        train_std = np.std(all_train_accs, axis=0)
        val_mean = np.mean(all_val_accs, axis=0)
        val_std = np.std(all_val_accs, axis=0)
        
        color = colors.get(opt_name, '#95a5a6')
        
        # Train Accuracy
        axes[0].plot(epochs, train_mean, label=opt_name, color=color, linewidth=2)
        axes[0].fill_between(epochs, train_mean - train_std, train_mean + train_std,
                             alpha=0.2, color=color)
        
        # Val Accuracy
        axes[1].plot(epochs, val_mean, label=opt_name, color=color, linewidth=2)
        axes[1].fill_between(epochs, val_mean - val_std, val_mean + val_std,
                             alpha=0.2, color=color)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Train Accuracy (%)', fontsize=12)
    axes[0].set_title('Training Accuracy vs Epoch', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[1].set_title('Validation Accuracy vs Epoch', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {os.path.join(output_dir, 'accuracy_curves.png')}")


def plot_loss_curves(
    grouped_metrics: Dict[str, List[Dict]],
    output_dir: str = "results"
):
    """绘制损失曲线图"""
    os.makedirs(output_dir, exist_ok=True)
    
    colors = {'SGD': '#2ecc71', 'AdamW': '#3498db', 'Muon': '#e74c3c'}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for opt_name, metrics_list in grouped_metrics.items():
        all_train_losses = []
        all_val_losses = []
        epochs = None
        
        for metrics in metrics_list:
            epoch_metrics = metrics['epoch_metrics']
            epochs = [em['epoch'] for em in epoch_metrics]
            train_losses = [em['train_loss'] for em in epoch_metrics]
            val_losses = [em['val_loss'] for em in epoch_metrics]
            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
        
        if epochs is None:
            continue
        
        train_mean = np.mean(all_train_losses, axis=0)
        train_std = np.std(all_train_losses, axis=0)
        val_mean = np.mean(all_val_losses, axis=0)
        val_std = np.std(all_val_losses, axis=0)
        
        color = colors.get(opt_name, '#95a5a6')
        
        axes[0].plot(epochs, train_mean, label=opt_name, color=color, linewidth=2)
        axes[0].fill_between(epochs, train_mean - train_std, train_mean + train_std,
                             alpha=0.2, color=color)
        
        axes[1].plot(epochs, val_mean, label=opt_name, color=color, linewidth=2)
        axes[1].fill_between(epochs, val_mean - val_std, val_mean + val_std,
                             alpha=0.2, color=color)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Train Loss', fontsize=12)
    axes[0].set_title('Training Loss vs Epoch', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].set_title('Validation Loss vs Epoch', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {os.path.join(output_dir, 'loss_curves.png')}")


def plot_time_comparison(
    grouped_metrics: Dict[str, List[Dict]],
    output_dir: str = "results"
):
    """绘制时间对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    colors = {'SGD': '#2ecc71', 'AdamW': '#3498db', 'Muon': '#e74c3c'}
    optimizers = ['SGD', 'AdamW', 'Muon']
    
    # 收集数据
    avg_step_times = {opt: [] for opt in optimizers}
    total_times = {opt: [] for opt in optimizers}
    
    for opt_name in optimizers:
        if opt_name in grouped_metrics:
            for m in grouped_metrics[opt_name]:
                avg_step_times[opt_name].append(m['avg_step_time_ms'])
                total_times[opt_name].append(m['total_train_time_s'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 平均 Step Time
    x = np.arange(len(optimizers))
    width = 0.6
    
    means = [np.mean(avg_step_times[opt]) if avg_step_times[opt] else 0 for opt in optimizers]
    stds = [np.std(avg_step_times[opt]) if avg_step_times[opt] else 0 for opt in optimizers]
    bars_colors = [colors[opt] for opt in optimizers]
    
    bars = axes[0].bar(x, means, width, yerr=stds, color=bars_colors, 
                       capsize=5, edgecolor='black', linewidth=1)
    axes[0].set_xlabel('Optimizer', fontsize=12)
    axes[0].set_ylabel('Avg Step Time (ms)', fontsize=12)
    axes[0].set_title('Average Step Time Comparison', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(optimizers, fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, mean, std in zip(bars, means, stds):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                     f'{mean:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 总训练时间
    means_total = [np.mean(total_times[opt]) if total_times[opt] else 0 for opt in optimizers]
    stds_total = [np.std(total_times[opt]) if total_times[opt] else 0 for opt in optimizers]
    
    bars = axes[1].bar(x, means_total, width, yerr=stds_total, color=bars_colors,
                       capsize=5, edgecolor='black', linewidth=1)
    axes[1].set_xlabel('Optimizer', fontsize=12)
    axes[1].set_ylabel('Total Training Time (s)', fontsize=12)
    axes[1].set_title('Total Training Time Comparison', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(optimizers, fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, mean, std in zip(bars, means_total, stds_total):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                     f'{mean:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {os.path.join(output_dir, 'time_comparison.png')}")


def plot_memory_comparison(
    grouped_metrics: Dict[str, List[Dict]],
    output_dir: str = "results"
):
    """绘制显存对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    colors = {'SGD': '#2ecc71', 'AdamW': '#3498db', 'Muon': '#e74c3c'}
    optimizers = ['SGD', 'AdamW', 'Muon']
    
    peak_mems = {opt: [] for opt in optimizers}
    gpu_utils = {opt: [] for opt in optimizers}
    
    for opt_name in optimizers:
        if opt_name in grouped_metrics:
            for m in grouped_metrics[opt_name]:
                peak_mems[opt_name].append(m['peak_mem_allocated_mb'])
                gpu_utils[opt_name].append(m['gpu_util_mean'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(optimizers))
    width = 0.6
    bars_colors = [colors[opt] for opt in optimizers]
    
    # 峰值显存
    means = [np.mean(peak_mems[opt]) if peak_mems[opt] else 0 for opt in optimizers]
    stds = [np.std(peak_mems[opt]) if peak_mems[opt] else 0 for opt in optimizers]
    
    bars = axes[0].bar(x, means, width, yerr=stds, color=bars_colors,
                       capsize=5, edgecolor='black', linewidth=1)
    axes[0].set_xlabel('Optimizer', fontsize=12)
    axes[0].set_ylabel('Peak Memory (MB)', fontsize=12)
    axes[0].set_title('Peak Memory Comparison', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(optimizers, fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar, mean, std in zip(bars, means, stds):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                     f'{mean:.1f}', ha='center', va='bottom', fontsize=10)
    
    # GPU 利用率
    means_gpu = [np.mean(gpu_utils[opt]) if gpu_utils[opt] else 0 for opt in optimizers]
    stds_gpu = [np.std(gpu_utils[opt]) if gpu_utils[opt] else 0 for opt in optimizers]
    
    bars = axes[1].bar(x, means_gpu, width, yerr=stds_gpu, color=bars_colors,
                       capsize=5, edgecolor='black', linewidth=1)
    axes[1].set_xlabel('Optimizer', fontsize=12)
    axes[1].set_ylabel('GPU Utilization (%)', fontsize=12)
    axes[1].set_title('GPU Utilization Comparison', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(optimizers, fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(0, 100)
    
    for bar, mean, std in zip(bars, means_gpu, stds_gpu):
        if mean > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2,
                         f'{mean:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {os.path.join(output_dir, 'memory_comparison.png')}")


def generate_report(
    grouped_metrics: Dict[str, List[Dict]],
    output_dir: str = "results"
):
    """生成实验报告 Markdown"""
    os.makedirs(output_dir, exist_ok=True)
    
    optimizers = ['SGD', 'AdamW', 'Muon']
    
    # 计算汇总统计
    summary = {}
    for opt_name in optimizers:
        if opt_name not in grouped_metrics:
            continue
        
        metrics_list = grouped_metrics[opt_name]
        summary[opt_name] = {
            'best_val_acc': (
                np.mean([m['best_val_acc'] for m in metrics_list]),
                np.std([m['best_val_acc'] for m in metrics_list])
            ),
            'best_val_epoch': [m['best_val_epoch'] for m in metrics_list],
            'test_acc_best': (
                np.mean([m['test_acc_best'] for m in metrics_list]),
                np.std([m['test_acc_best'] for m in metrics_list])
            ),
            'test_acc_last': (
                np.mean([m['test_acc_last'] for m in metrics_list]),
                np.std([m['test_acc_last'] for m in metrics_list])
            ),
            'avg_step_time': (
                np.mean([m['avg_step_time_ms'] for m in metrics_list]),
                np.std([m['avg_step_time_ms'] for m in metrics_list])
            ),
            'peak_mem': (
                np.mean([m['peak_mem_allocated_mb'] for m in metrics_list]),
                np.std([m['peak_mem_allocated_mb'] for m in metrics_list])
            ),
            'gpu_util': (
                np.mean([m['gpu_util_mean'] for m in metrics_list]),
                np.std([m['gpu_util_mean'] for m in metrics_list])
            ),
        }
    
    # 生成 Markdown 报告
    report = []
    report.append("# LeNet 多优化器对比实验报告\n")
    report.append(f"生成时间: {np.datetime64('now')}\n")
    
    report.append("\n## 1. 实验配置\n")
    report.append("- 模型: LeNet")
    report.append("- 数据集: CIFAR-10")
    report.append("- Batch Size: 128")
    report.append("- Epochs: 30")
    report.append("- 每个优化器运行: 3 次（不同随机种子）\n")
    
    report.append("\n## 2. 优化器参数配置\n")
    report.append("| 优化器 | LR | Weight Decay | Momentum | 其他 |")
    report.append("|--------|------|--------------|----------|------|")
    report.append("| SGD | 0.05 | 1e-4 | 0.9 | Nesterov=True |")
    report.append("| AdamW | 0.001 | 0.01 | β1=0.9, β2=0.999 | ε=1e-8 |")
    report.append("| Muon | 0.002 | 0.01 | 0.95 | Nesterov=True, ns_steps=5 |\n")
    
    report.append("\n## 3. 最终性能指标表\n")
    report.append("| Optimizer | Best Val Acc | Epoch@Best | Test Acc (Best-Val) | Test Acc (Last) | Avg Step Time (ms) | Peak Mem (MB) | GPU Util Mean |")
    report.append("|-----------|-------------|------------|---------------------|-----------------|-------------------|---------------|---------------|")
    
    for opt_name in optimizers:
        if opt_name not in summary:
            continue
        s = summary[opt_name]
        epochs_str = ', '.join(map(str, s['best_val_epoch']))
        report.append(
            f"| {opt_name} | "
            f"{s['best_val_acc'][0]:.2f} ± {s['best_val_acc'][1]:.2f} | "
            f"{epochs_str} | "
            f"{s['test_acc_best'][0]:.2f} ± {s['test_acc_best'][1]:.2f} | "
            f"{s['test_acc_last'][0]:.2f} ± {s['test_acc_last'][1]:.2f} | "
            f"{s['avg_step_time'][0]:.2f} ± {s['avg_step_time'][1]:.2f} | "
            f"{s['peak_mem'][0]:.1f} ± {s['peak_mem'][1]:.1f} | "
            f"{s['gpu_util'][0]:.1f} ± {s['gpu_util'][1]:.1f} |"
        )
    
    report.append("\n\n## 4. 准确率曲线\n")
    report.append("![Accuracy Curves](accuracy_curves.png)\n")
    
    report.append("\n## 5. 损失曲线\n")
    report.append("![Loss Curves](loss_curves.png)\n")
    
    report.append("\n## 6. 时间对比\n")
    report.append("![Time Comparison](time_comparison.png)\n")
    
    report.append("\n## 7. 显存与GPU利用率对比\n")
    report.append("![Memory Comparison](memory_comparison.png)\n")
    
    # 保存报告
    report_path = os.path.join(output_dir, 'experiment_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Saved: {report_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory containing log files")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for figures and report")
    args = parser.parse_args()
    
    print("Loading metrics...")
    grouped_metrics = load_all_metrics(args.log_dir)
    
    if not grouped_metrics:
        print("No metrics files found!")
        return
    
    print(f"Found metrics for: {list(grouped_metrics.keys())}")
    
    print("\nGenerating plots...")
    plot_accuracy_curves(grouped_metrics, args.output_dir)
    plot_loss_curves(grouped_metrics, args.output_dir)
    plot_time_comparison(grouped_metrics, args.output_dir)
    plot_memory_comparison(grouped_metrics, args.output_dir)
    
    print("\nGenerating report...")
    generate_report(grouped_metrics, args.output_dir)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

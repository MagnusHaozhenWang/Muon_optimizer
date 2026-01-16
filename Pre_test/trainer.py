"""
训练器模块 - 封装训练、评估、checkpoint 等功能
"""
import os
import csv
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from utils import StepTimer, GPUMonitor, get_memory_stats, reset_memory_stats


@dataclass
class EpochMetrics:
    """单个 epoch 的指标"""
    epoch: int
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    epoch_time_s: float = 0.0
    lr: float = 0.0


@dataclass
class RunMetrics:
    """完整运行的指标"""
    run_id: str
    optimizer: str
    seed: int
    epoch_metrics: List[EpochMetrics] = field(default_factory=list)
    
    # 最终指标
    best_val_acc: float = 0.0
    best_val_epoch: int = 0
    test_acc_best: float = 0.0
    test_acc_last: float = 0.0
    
    # 时间指标
    avg_step_time_ms: float = 0.0
    p50_step_time_ms: float = 0.0
    p90_step_time_ms: float = 0.0
    total_train_time_s: float = 0.0
    
    # GPU 指标
    peak_mem_allocated_mb: float = 0.0
    peak_mem_reserved_mb: float = 0.0
    gpu_util_mean: float = 0.0
    gpu_util_p50: float = 0.0
    gpu_util_p90: float = 0.0


class Trainer:
    """训练器类"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
        run_id: str = "",
        optimizer_name: str = "",
        seed: int = 0,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        warmup_steps: int = 50,
        gpu_sample_interval_ms: int = 200,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.run_id = run_id
        self.optimizer_name = optimizer_name
        self.seed = seed
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        # 时间统计
        self.step_timer = StepTimer(warmup_steps=warmup_steps)
        
        # GPU 监控
        self.gpu_monitor = GPUMonitor(sample_interval_ms=gpu_sample_interval_ms)
        
        # 运行指标
        self.metrics = RunMetrics(
            run_id=run_id,
            optimizer=optimizer_name,
            seed=seed,
        )
        
        # 日志文件
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{run_id}.csv")
        self._init_log_file()
    
    def _init_log_file(self):
        """初始化日志文件"""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'run_id', 'seed', 'optimizer', 'epoch', 'split',
                'loss', 'acc', 'lr', 'epoch_time_s', 'timestamp'
            ])
    
    def _log_epoch(self, epoch: int, split: str, loss: float, acc: float, 
                   epoch_time: float = 0.0):
        """记录 epoch 日志"""
        lr = self.optimizer.param_groups[0]['lr']
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.run_id, self.seed, self.optimizer_name, epoch, split,
                f"{loss:.6f}", f"{acc:.4f}", f"{lr:.6f}", 
                f"{epoch_time:.2f}", datetime.now().isoformat()
            ])
    
    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.step_timer.start()
            
            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 优化器更新
            self.optimizer.step()
            
            self.step_timer.stop()
            
            # 统计
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'run_id': self.run_id,
        }
        
        # 保存最新
        path = os.path.join(self.checkpoint_dir, f"{self.run_id}_last.pth")
        torch.save(checkpoint, path)
        
        # 保存最佳
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"{self.run_id}_best.pth")
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['val_acc']
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        eval_freq: int = 5,
    ) -> RunMetrics:
        """完整训练流程"""
        
        # 重置统计
        reset_memory_stats()
        self.step_timer.reset()
        
        # 启动 GPU 监控
        self.gpu_monitor.start()
        
        best_val_acc = 0.0
        best_val_epoch = 0
        total_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Training: {self.run_id}")
        print(f"Optimizer: {self.optimizer_name}, Seed: {self.seed}")
        print(f"{'='*60}")
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)
            
            epoch_time = time.time() - epoch_start
            
            # 每 eval_freq 个 epoch 评估
            if epoch % eval_freq == 0 or epoch == epochs:
                val_loss, val_acc = self.evaluate(val_loader)
                
                # 记录指标
                epoch_metrics = EpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    epoch_time_s=epoch_time,
                    lr=self.optimizer.param_groups[0]['lr'],
                )
                self.metrics.epoch_metrics.append(epoch_metrics)
                
                # 日志
                self._log_epoch(epoch, 'train', train_loss, train_acc, epoch_time)
                self._log_epoch(epoch, 'val', val_loss, val_acc)
                
                # 更新最佳
                is_best = val_acc > best_val_acc
                if is_best:
                    best_val_acc = val_acc
                    best_val_epoch = epoch
                
                # 保存检查点
                self.save_checkpoint(epoch, val_acc, is_best)
                
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
                      f"Time: {epoch_time:.1f}s")
        
        total_train_time = time.time() - total_start_time
        
        # 停止 GPU 监控
        gpu_stats = self.gpu_monitor.stop()
        
        # 测试集评估
        # Last epoch
        test_loss_last, test_acc_last = self.evaluate(test_loader)
        
        # Best val checkpoint
        best_ckpt_path = os.path.join(self.checkpoint_dir, f"{self.run_id}_best.pth")
        if os.path.exists(best_ckpt_path):
            self.load_checkpoint(best_ckpt_path)
        test_loss_best, test_acc_best = self.evaluate(test_loader)
        
        # 记录最终指标
        step_stats = self.step_timer.get_stats()
        mem_stats = get_memory_stats()
        
        self.metrics.best_val_acc = best_val_acc
        self.metrics.best_val_epoch = best_val_epoch
        self.metrics.test_acc_best = test_acc_best
        self.metrics.test_acc_last = test_acc_last
        
        self.metrics.avg_step_time_ms = step_stats['mean']
        self.metrics.p50_step_time_ms = step_stats['p50']
        self.metrics.p90_step_time_ms = step_stats['p90']
        self.metrics.total_train_time_s = total_train_time
        
        self.metrics.peak_mem_allocated_mb = mem_stats['allocated_mb']
        self.metrics.peak_mem_reserved_mb = mem_stats['reserved_mb']
        self.metrics.gpu_util_mean = gpu_stats['mean']
        self.metrics.gpu_util_p50 = gpu_stats['p50']
        self.metrics.gpu_util_p90 = gpu_stats['p90']
        
        # 日志测试结果
        self._log_epoch(epochs, 'test_last', test_loss_last, test_acc_last)
        self._log_epoch(epochs, 'test_best', test_loss_best, test_acc_best)
        
        print(f"\n{'='*60}")
        print(f"Training Complete: {self.run_id}")
        print(f"Best Val Acc: {best_val_acc:.2f}% @ Epoch {best_val_epoch}")
        print(f"Test Acc (Best-Val): {test_acc_best:.2f}%")
        print(f"Test Acc (Last): {test_acc_last:.2f}%")
        print(f"Avg Step Time: {step_stats['mean']:.2f} ms")
        print(f"Peak Memory: {mem_stats['allocated_mb']:.1f} MB")
        print(f"Total Time: {total_train_time:.1f}s")
        print(f"{'='*60}\n")
        
        # 保存完整指标到 JSON
        metrics_path = os.path.join(self.log_dir, f"{self.run_id}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(asdict(self.metrics), f, indent=2)
        
        return self.metrics

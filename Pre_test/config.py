"""
配置管理模块 - LeNet 多优化器对比实验
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class TrainConfig:
    """训练配置"""
    batch_size: int = 512
    epochs: int = 100
    eval_freq: int = 5  # 每 5 个 epoch 评估一次
    warmup_steps: int = 50  # step 时间测量时跳过的 warmup 步数
    
    # 数据集划分
    val_size: int = 5000  # 从训练集划分的验证集大小
    data_split_seed: int = 42  # 固定数据划分的种子
    
    # 随机种子列表（每个优化器运行 3 次）
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    
    # 输出目录
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    result_dir: str = "results"
    
    # GPU 监控
    gpu_sample_interval_ms: int = 200  # GPU 利用率采样间隔


@dataclass
class SGDConfig:
    """SGD 优化器配置"""
    name: str = "SGD"
    lr: float = 0.04
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = True


@dataclass
class AdamWConfig:
    """AdamW 优化器配置"""
    name: str = "AdamW"
    lr: float = 0.004
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class MuonConfig:
    """Muon 优化器配置"""
    name: str = "Muon"
    lr: float = 0.005
    momentum: float = 0.95
    weight_decay: float = 0.01
    nesterov: bool = True
    ns_steps: int = 5  # Newton-Schulz 迭代步数


# CIFAR-10 归一化参数
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_optimizer_configs() -> Dict[str, Any]:
    """获取所有优化器配置"""
    return {
        "SGD": SGDConfig(),
        "AdamW": AdamWConfig(),
        "Muon": MuonConfig(),
    }


def get_train_config() -> TrainConfig:
    """获取训练配置"""
    return TrainConfig()

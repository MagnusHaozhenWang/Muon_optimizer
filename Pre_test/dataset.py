"""
数据集模块 - CIFAR-10 数据加载与划分
"""
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from typing import Tuple
import numpy as np

from config import CIFAR10_MEAN, CIFAR10_STD, TrainConfig
from utils import worker_init_fn


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    获取训练和测试的数据变换
    
    Returns:
        (train_transform, test_transform)
    """
    # 训练集变换（不使用数据增强，保持控制变量）
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    
    # 测试/验证集变换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    
    return train_transform, test_transform


def get_dataloaders(
    cfg: TrainConfig,
    data_dir: str = "./data"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    获取 CIFAR-10 数据加载器
    
    Args:
        cfg: 训练配置
        data_dir: 数据存储目录
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_transform, test_transform = get_transforms()
    
    # 下载/加载完整训练集
    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    
    # 固定种子划分 train/val
    generator = torch.Generator().manual_seed(cfg.data_split_seed)
    train_size = len(full_train_dataset) - cfg.val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, cfg.val_size],
        generator=generator,
    )
    
    # 验证集使用测试变换（虽然这里相同，但保持语义正确）
    # 由于 random_split 返回的 Subset 共享 transform，这里重新包装
    val_dataset.dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=test_transform,
    )
    
    # 测试集
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(cfg.data_split_seed),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def get_dataset_info() -> dict:
    """获取数据集信息"""
    return {
        "name": "CIFAR-10",
        "num_classes": 10,
        "image_size": (32, 32, 3),
        "train_size": 45000,  # 50000 - 5000 val
        "val_size": 5000,
        "test_size": 10000,
        "normalize_mean": CIFAR10_MEAN,
        "normalize_std": CIFAR10_STD,
    }

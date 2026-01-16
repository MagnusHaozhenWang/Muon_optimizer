"""
优化器工厂模块 - 根据配置创建优化器
"""
import torch
import torch.optim as optim
from typing import Any, Iterator
from torch.nn import Parameter

from muon import Muon
from config import SGDConfig, AdamWConfig, MuonConfig


def build_optimizer(
    name: str,
    params: Iterator[Parameter],
    cfg: Any
) -> optim.Optimizer:
    """
    优化器工厂函数
    
    Args:
        name: 优化器名称 (SGD, AdamW, Muon)
        params: 模型参数迭代器
        cfg: 优化器配置对象
    
    Returns:
        对应的优化器实例
    """
    name = name.lower()
    
    if name == "sgd":
        assert isinstance(cfg, SGDConfig)
        return optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=cfg.nesterov,
        )
    
    elif name == "adamw":
        assert isinstance(cfg, AdamWConfig)
        return optim.AdamW(
            params,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
    
    elif name == "muon":
        assert isinstance(cfg, MuonConfig)
        return Muon(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay,
            ns_steps=cfg.ns_steps,
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def get_optimizer_info(optimizer: optim.Optimizer) -> dict:
    """获取优化器信息"""
    info = {
        "name": type(optimizer).__name__,
        "param_groups": len(optimizer.param_groups),
    }
    
    # 添加第一个 param group 的超参数
    if optimizer.param_groups:
        pg = optimizer.param_groups[0]
        for key in ['lr', 'momentum', 'weight_decay', 'betas', 'eps', 'nesterov', 'ns_steps']:
            if key in pg:
                info[key] = pg[key]
    
    return info

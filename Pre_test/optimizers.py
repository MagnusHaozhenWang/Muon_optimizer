"""
优化器工厂模块 - 根据配置创建优化器
"""
import torch
import torch.optim as optim
from typing import Any, Iterator, List
from torch.nn import Parameter, Module

from muon import Muon
from config import SGDConfig, AdamWConfig, MuonConfig


class CombinedOptimizer:
    """
    组合优化器 - 封装多个优化器实现统一接口
    
    用于 Muon 优化器：
    - 2D 参数（卷积层权重、全连接层权重）使用 Muon
    - 非 2D 参数（bias、BatchNorm 等）使用 AdamW
    """
    
    def __init__(self, optimizers: List[optim.Optimizer]):
        self.optimizers = optimizers
        # 创建统一的 param_groups 视图（用于学习率调度等）
        self._param_groups = []
        for opt in optimizers:
            self._param_groups.extend(opt.param_groups)
    
    @property
    def param_groups(self):
        """返回所有优化器的 param_groups"""
        return self._param_groups
    
    def zero_grad(self, set_to_none: bool = True):
        """清零所有优化器的梯度"""
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)
    
    def step(self, closure=None):
        """执行所有优化器的更新步骤"""
        loss = None
        for opt in self.optimizers:
            result = opt.step(closure)
            if result is not None:
                loss = result
        return loss
    
    def state_dict(self):
        """返回所有优化器的状态字典"""
        return {
            'optimizers': [opt.state_dict() for opt in self.optimizers]
        }
    
    def load_state_dict(self, state_dict):
        """加载所有优化器的状态字典"""
        for opt, opt_state in zip(self.optimizers, state_dict['optimizers']):
            opt.load_state_dict(opt_state)


def build_optimizer(
    name: str,
    params: Iterator[Parameter],
    cfg: Any,
    model: Module = None,
) -> optim.Optimizer:
    """
    优化器工厂函数
    
    Args:
        name: 优化器名称 (SGD, AdamW, Muon)
        params: 模型参数迭代器 (对于 Muon 会被忽略，使用 model 参数)
        cfg: 优化器配置对象
        model: 模型实例 (仅 Muon 需要，用于分离 2D 和非 2D 参数)
    
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
        
        if model is None:
            # 向后兼容：如果没有提供 model，使用原来的方式
            return Muon(
                params,
                lr=cfg.lr,
                momentum=cfg.momentum,
                nesterov=cfg.nesterov,
                weight_decay=cfg.weight_decay,
                ns_steps=cfg.ns_steps,
            )
        
        # 分离 2D 参数和非 2D 参数
        muon_params = []  # 2D 参数使用 Muon
        adamw_params = []  # 非 2D 参数使用 AdamW
        
        for name_p, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 2D 参数（卷积层权重、全连接层权重）使用 Muon
            if param.ndim >= 2:
                muon_params.append(param)
            else:
                # 非 2D 参数（bias、BatchNorm 参数等）使用 AdamW
                adamw_params.append(param)
        
        optimizers = []
        
        # 创建 Muon 优化器（用于 2D 参数）
        if muon_params:
            muon_opt = Muon(
                muon_params,
                lr=cfg.lr,
                momentum=cfg.momentum,
                nesterov=cfg.nesterov,
                weight_decay=cfg.weight_decay,
                ns_steps=cfg.ns_steps,
            )
            optimizers.append(muon_opt)
        
        # 创建 AdamW 优化器（用于非 2D 参数）
        if adamw_params:
            adamw_opt = optim.AdamW(
                adamw_params,
                lr=cfg.lr,  # 使用相同的学习率
                weight_decay=cfg.weight_decay,
            )
            optimizers.append(adamw_opt)
        
        if len(optimizers) == 1:
            return optimizers[0]
        
        return CombinedOptimizer(optimizers)
    
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def get_optimizer_info(optimizer) -> dict:
    """获取优化器信息"""
    if isinstance(optimizer, CombinedOptimizer):
        return {
            "name": "CombinedOptimizer(Muon+AdamW)",
            "param_groups": len(optimizer.param_groups),
            "sub_optimizers": [type(opt).__name__ for opt in optimizer.optimizers],
        }
    
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


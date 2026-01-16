"""
Muon 优化器实现 - Momentum Orthogonal Optimizer
基于 Newton-Schulz 迭代进行梯度正交化
"""
import torch
from torch.optim import Optimizer
from typing import List, Optional, Callable


def newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz 迭代计算矩阵的近似正交化
    
    Args:
        G: 输入梯度矩阵
        steps: 迭代步数
        eps: 数值稳定性参数
    
    Returns:
        正交化后的矩阵
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    
    if G.size(0) > G.size(1):
        X = X.T
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    
    return X


class Muon(Optimizer):
    """
    Muon (Momentum Orthogonal) 优化器
    
    使用 Newton-Schulz 迭代对梯度进行正交化处理，
    结合动量更新实现高效优化。
    
    Args:
        params: 模型参数
        lr: 学习率 (默认: 0.002)
        momentum: 动量系数 (默认: 0.95)
        nesterov: 是否使用 Nesterov 动量 (默认: True)
        weight_decay: 权重衰减 (默认: 0.01)
        ns_steps: Newton-Schulz 迭代步数 (默认: 5)
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.002,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super(Muon, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """执行单步优化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # 权重衰减 (decoupled weight decay, 类似 AdamW)
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
                
                # 对于 2D 参数（如全连接层权重），应用正交化
                if grad.ndim == 2:
                    # Newton-Schulz 正交化
                    grad_orth = newtonschulz5(grad, steps=ns_steps)
                    grad = grad_orth.to(grad.dtype)
                
                # 获取或初始化动量缓冲区
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)
                
                # Nesterov 或普通动量
                if nesterov:
                    grad = grad.add(buf, alpha=momentum)
                else:
                    grad = buf
                
                # 参数更新
                p.data.add_(grad, alpha=-lr)
        
        return loss

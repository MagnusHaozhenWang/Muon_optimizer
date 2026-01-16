"""
工具函数模块 - 随机种子、GPU监控等
"""
import os
import random
import threading
import time
from typing import List, Optional, Dict
from dataclasses import dataclass, field

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保确定性（可能降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker 的种子初始化函数"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_system_info() -> Dict[str, str]:
    """获取系统信息"""
    import platform
    info = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.cuda.is_available() else "N/A",
        "platform": platform.platform(),
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}"
    
    return info


class GPUMonitor:
    """后台 GPU 利用率监控器"""
    
    def __init__(self, sample_interval_ms: int = 200):
        self.sample_interval = sample_interval_ms / 1000.0  # 转换为秒
        self.samples: List[float] = []
        self.memory_samples: List[float] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pynvml_available = False
        
        # 尝试导入 pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml_available = True
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            print("Warning: pynvml not available, GPU utilization monitoring disabled")
    
    def start(self) -> None:
        """开始监控"""
        if not self._pynvml_available:
            return
        self._running = True
        self.samples = []
        self.memory_samples = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> Dict[str, float]:
        """停止监控并返回统计结果"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        if not self.samples:
            return {"mean": 0, "p50": 0, "p90": 0}
        
        samples = np.array(self.samples)
        return {
            "mean": float(np.mean(samples)),
            "p50": float(np.percentile(samples, 50)),
            "p90": float(np.percentile(samples, 90)),
        }
    
    def _sample_loop(self) -> None:
        """采样循环"""
        import pynvml
        while self._running:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self.samples.append(util.gpu)
                self.memory_samples.append(util.memory)
            except Exception:
                pass
            time.sleep(self.sample_interval)


@dataclass
class StepTimer:
    """Step 时间统计器"""
    warmup_steps: int = 50
    step_times: List[float] = field(default_factory=list)
    _current_step: int = 0
    _start_time: float = 0.0
    
    def start(self) -> None:
        """开始计时"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()
    
    def stop(self) -> Optional[float]:
        """停止计时并记录"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - self._start_time) * 1000  # 转换为毫秒
        
        self._current_step += 1
        if self._current_step > self.warmup_steps:
            self.step_times.append(elapsed)
            return elapsed
        return None
    
    def get_stats(self) -> Dict[str, float]:
        """获取统计结果"""
        if not self.step_times:
            return {"mean": 0, "p50": 0, "p90": 0}
        
        times = np.array(self.step_times)
        return {
            "mean": float(np.mean(times)),
            "p50": float(np.percentile(times, 50)),
            "p90": float(np.percentile(times, 90)),
        }
    
    def reset(self) -> None:
        """重置计时器"""
        self.step_times = []
        self._current_step = 0


def get_memory_stats() -> Dict[str, float]:
    """获取 GPU 显存统计"""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0}
    
    return {
        "allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        "reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
    }


def reset_memory_stats() -> None:
    """重置显存统计"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

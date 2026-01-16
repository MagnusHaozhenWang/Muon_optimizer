# LeNet 多优化器对比实验方案（SGD / AdamW / Muon）

## 1. 实验目标

- 对比 **SGD、AdamW、Muon(Momentum Orthogonal Optimizer)** 三种优化器在 **LeNet** 模型上的训练效果差异。
- 评估维度：
  - 训练集准确率（Train Acc）
  - 验证集准确率（Val Acc）
  - 收敛速度（Convergence Speed）
  - 训练时间（Time）
  - GPU 占用（显存、利用率）

## 2. 指标定义（统一口径）

### 2.1 准确率与损失

- Train Acc / Val Acc / Test Acc：Top-1 accuracy。
- Train Loss / Val Loss：交叉熵损失的平均值。

### 2.2 收敛速度（至少采用一种，推荐两种都报告）

- **阈值收敛**：达到目标验证准确率阈值所需的 epoch / step / 时间（例如 `Val Acc ≥ 98%`，阈值可按数据集难度调整）。
- **最佳验证性能**：最佳 Val Acc 及其出现的 epoch（并记录对应 checkpoint）。

### 2.3 时间指标

- **平均 step 时间**：训练阶段单次迭代（forward + backward + optimizer.step）的平均耗时（去除 warmup）。
- **epoch 时间**：每个 epoch 的训练耗时（可选：与验证耗时分开统计）。
- **总训练时间**：从第一步训练开始到最后一步训练结束的累计时间（需明确是否包含验证与保存开销）。

### 2.4 GPU 指标

- **显存峰值**：`max_memory_allocated` / `max_memory_reserved`（MB）。
- **GPU 利用率**：在训练过程中以固定间隔采样（如 200ms），统计 mean / P50 / P90（%）。

## 3. 实验环境配置（可复现性要求）

### 3.1 软件与硬件记录

- 记录以下信息到日志与报告中：
  - Python 版本、PyTorch 版本、CUDA/cuDNN 版本、驱动版本
  - GPU 型号与显存大小、CPU 型号、内存大小

### 3.2 随机种子与确定性策略

- 固定随机种子：`random`、`numpy`、`torch`、`torch.cuda`。
- 固定 DataLoader worker 的种子（例如在 `worker_init_fn` 中设置）。

### 3.3 重复实验与统计

- 每个优化器至少运行 **N=3** 个不同随机种子。
- 对最终 Test Acc、收敛速度、平均 step 时间、GPU 指标报告 **均值 ± 标准差**。

## 4. 数据集与划分

### 4.1 数据集

- 使用同一数据集与同一预处理策略，选用：
  - CIFAR-10

### 4.2 数据划分（固定）

- 固定 train/val/test 划分方式：
  - 例如从训练集划分 `K` 条样本作为验证集（如 5,000），剩余作为训练集。
- 固定划分 seed，确保三种优化器使用完全相同的数据划分与 DataLoader 设置。

### 4.3 预处理

- Normalize 参数固定（写入配置）。
- 是否使用数据增强需明确。

## 5. 模型与训练设置（控制变量）

### 5.1 模型

- 使用 PyTorch 实现 LeNet，并确保三种优化器使用：
  - 完全一致的模型结构
  - 完全一致的权重初始化方式
  - 完全一致的初始权重（同 seed 或直接复用同一初始化权重）

### 5.2 训练超参数（固定）

- batch size：固定（示例：128）
- epoch 数：固定（示例：30 或 50）
- 损失函数：`CrossEntropyLoss`
- 评估频率：每 5 个 epoch 记录一次 train/val 指标（可额外每个 epoch 记录 loss 以便分析）
- 其他训练细节必须固定并声明：
  - 是否使用学习率调度器（建议先不使用；若使用必须全优化器一致）
  - 是否使用 weight decay / 梯度裁剪 / label smoothing / AMP（混合精度）

## 6. 优化器设置

### 6.1 参数配置表（需在报告中完整列出）
| **优化器 (Optimizer)** | **学习率 (LR)** | **Weight Decay** | **Momentum (β1​)** | **二阶/其他参数**                                   | **关键说明**                                         |
| ------------------- | ------------ | ---------------- | ------------------ | --------------------------------------------- | ------------------------------------------------ |
| **SGD**             | **0.05**     | 1e-4             | **0.9**            | Nesterov=True                                 | 必须开启 Momentum 和 Nesterov                         |
| **AdamW**           | **0.001**    | **0.01**         | 0.9                | $\beta_2$=0.999, $\epsilon$=1e-8              | **强烈建议用 AdamW 代替 Adam**。AdamW 的 WD 实现是正确的，通用性更强。 |
| **Muon**            | **0.002**    | 0.01             | **0.95**           | Nesterov=True<br><br>  <br><br>**ns_steps=5** | Muon 喜欢较大的 Momentum。LR 通常比 Adam 大，但比 SGD 小。      |

Muon示例代码
```python
# Pytorch code
def newtonschulz5(G, steps=5, eps=1e-7):
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
```
### 6.2 学习率合理性校验（建议加入，避免“lr 没配好”的争议）

- 在正式训练前，对每个优化器做一次短跑 sanity check：
  - 学习率候选：`{0.1×, 1×, 10×}`（即 3 个 lr）
  - 训练预算：例如 5 个 epoch
  - 选取：不发散且验证集表现最佳的 lr 进入正式对比
- 该流程保持同样的数据划分与种子，记录到日志（但不作为主对比结果）。

## 7. 训练与评估流程（统一流程）

### 7.1 训练循环要求

- 统一训练步骤：
  1) 前向传播
  2) 计算 loss
  3) 反向传播
  4) optimizer.step
  5) 记录 step 时间与显存信息

### 7.2 评估与记录频率

- 每 5 个 epoch：
  - 记录 train acc / train loss
  - 记录 val acc / val loss
- 每个 epoch（推荐）：
  - 记录 epoch 训练耗时
  - 更新 best-val checkpoint（若采用 best-val 口径）

### 7.3 测试集评估口径（必须统一并在报告中说明）

建议同时报告两种结果：

- **Last**：最后一个 epoch 的模型在测试集上的准确率。
- **Best-Val**：验证集最佳 checkpoint 在测试集上的准确率。

## 8. 时间与 GPU 占用测量方法（可执行口径）

### 8.1 step 时间测量

- 在 GPU 上计时时：
  - 先 warmup 若干 step（例如前 50 个 step 不计入统计）
  - 计时区间前后执行 `torch.cuda.synchronize()`
- 统计：
  - 平均 step time（ms）
  - P50 / P90 step time（ms）（可选但推荐）

### 8.2 显存统计

- 训练前清零峰值计数：
  - `torch.cuda.reset_peak_memory_stats()`
- 训练中/训练后记录：
  - `torch.cuda.max_memory_allocated()`（峰值 allocated）
  - `torch.cuda.max_memory_reserved()`（峰值 reserved）

### 8.3 GPU 利用率统计（推荐 NVML 采样）

- 以固定间隔（如 200ms）采样 GPU utilization 与 memory utilization。
- 汇总输出：
  - mean / P50 / P90
- 采样过程尽量放在独立线程/进程，避免对训练产生显著干扰。

## 9. 结果可视化与实验产物

### 9.1 学习曲线图（必做）

- 图 1：Train/Val Accuracy vs Epoch（每 5 epoch 一个点）
- 图 2：Train/Val Loss vs Epoch（建议同样绘制，便于解释收敛与过拟合）
- 若多 seed：绘制均值曲线并用阴影显示 ±std。

### 9.2 时间与资源图

- 条形图：不同优化器的平均 step time / epoch time / 总训练时间
- 条形图：显存峰值对比（allocated / reserved）
- 折线或箱线图：GPU utilization 分布对比（mean/P50/P90）

## 10. 实验报告结构

### 10.1 优化器参数配置表

- 按 §6.1 表格列出所有超参数（包含默认值与是否使用）。

### 10.2 准确率曲线对比图

- Accuracy vs Epoch（Train/Val）
- Loss vs Epoch（Train/Val）

### 10.3 时间对比数据

- 平均 step time、epoch time、总训练时间（均值±标准差）

### 10.4 最终性能指标表

| Optimizer | Best Val Acc | Epoch@Best | Test Acc (Best-Val) | Test Acc (Last) | Avg Step Time (ms) | Peak Mem (MB) | GPU Util Mean/P90 |
|---|---:|---:|---:|---:|---:|---:|---|
| SGD |  |  |  |  |  |  |  |
| Adam |  |  |  |  |  |  |  |
| Muon |  |  |  |  |  |  |  |

## 11. 代码实现建议（模块化、便于扩展）

### 11.1 模块划分

- 配置管理：集中管理训练超参、优化器超参、随机种子、输出目录。
- 模型定义：LeNet + 初始化函数。
- 优化器工厂：通过名称构建优化器，便于扩展新优化器。
- 训练器：封装 train_one_epoch / evaluate / checkpoint。
- 日志记录：输出结构化日志（CSV 或 JSONL），字段包含：
  - epoch、step、split(train/val/test)、loss、acc、lr、step_time_ms、epoch_time_s、mem_mb、gpu_util%
- 可视化：读取日志并输出曲线图与对比图。

### 11.2 优化器工厂接口（示例）

```python
def build_optimizer(name: str, params, cfg):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.nesterov, weight_decay=cfg.weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
    if name == "muon":
        return Muon(params, lr=cfg.lr, **cfg.muon_kwargs)
    raise ValueError(f"Unknown optimizer: {name}")
```

### 11.3 日志输出格式（建议字段）

- `run_id, seed, optimizer, epoch, step, split, loss, acc, lr, step_time_ms, epoch_time_s, peak_mem_mb, gpu_util_mean, timestamp`

## 12. 验收标准（完成一次对比实验的最小交付物）

- 三种优化器在相同设置下完整训练完成（不发散）
- 生成并保存：
  - 结构化日志文件（每个 run 一份）
  - Accuracy/Loss 曲线图
  - 时间与资源对比表
  - 最终性能指标表（含均值±标准差）


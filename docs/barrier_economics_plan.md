# V∞ Barrier Economics — 规划与架构设计

## 1. 为什么

Vinfty 当前对所有用户一视同仁：`active_k` 硬编码，ont_self 永远是全量 O(N²) 计算。但用户跑在树莓派上还是数据中心的 64 核服务器上，能承受的计算深度完全不同。

势垒经济学解决的就是这个问题：**给定 λ（算力相对认知的昂贵程度），自动选择最优计算深度和算法复杂度**。

| 场景 | λ | 推荐算子 | 对 vinfty 的影响 |
|------|---|---------|----------------|
| 数据中心推理 （A100） | 1e-5 | Fractal | 全量 9D 耦合 + 嵌套坍缩检测 |
| 个人 PC / 开发机 | 0.001 | ODE | 全量计算，默认 |
| 树莓派 / Edge 设备 | 0.1 | exp/log | 只算 Palace 级聚合，O(N) |
| MCU / 微控制器 | 100 | PRNG | 近似估计，qos=low |

## 2. 做什么

### 2.1 `vinfty/barrier.py` — 核心模块（~120 行）

从 `kernel/lambda_scheduler.py` 提取纯净版，剥离 T7Adapter、嵌套修正等 vinfty 不需要的东西：

```
vinfty/barrier/
├── __init__.py          # 导出 API（~5 行）
├── table.py             # OPERATOR_TABLE + compute_* 纯函数（~60 行）
├── estimator.py         # LambdaEstimator: 硬件→λ（~40 行）
└── scheduler.py         # LambdaScheduler + recommend()（~40 行）
```

`table.py` 不引入任何依赖——5 个算子的参数表和 3 个数学公式。

### 2.2 `V9Orchestrator` 集成

在 `core.py` 中新增 3 处改动：

```python
class V9Orchestrator:
    def __init__(self, active_k=100, lambda_estimate=None):
        # 新增
        self._barrier = LambdaScheduler(lam=lambda_estimate or auto_detect_lambda())
        self._mode = self._barrier.evaluate()["recommended_operator"]
```

自动 λ 检测（新增 `estimator.auto_detect()`）：

```python
def auto_detect() -> float:
    """Try platform-detection, fallback 0.001 (PC default)."""
    import platform, os
    # ... cpu_count, freq, mem
    return LambdaEstimator().estimate(...)
```

### 2.3 `step()` 自适应

```python
def step(self, content, tool=None, **extra):
    # 原有逻辑
    mem = {...}
    self.memories.append(mem)
    # 新增：根据当前算子决定计算深度
    if self._mode == "Fractal":
        self._compute_full_9d_coupling()
    elif self._mode == "ODE":
        self._compute_ont_self()  # 原有全量
    elif self._mode == "exp_log":
        self._compute_palace_aggregate()  # 只算 Palace 级
    elif self._mode == "PRNG":
        pass  # 跳过耦合计算
    self._scan_hmm()
```

### 2.4 `report()` 扩展

```python
def report(self) -> dict:
    base = {原有字段}
    base["barrier"] = {
        "lambda": self._barrier.lam,
        "mode": self._mode,
        "scenario": self._barrier._classify_scenario(self._barrier.lam),
        "collapse_risk": self._barrier.collapse_detector.update(...)["severity"],
    }
    return base
```

### 2.5 动态模式切换

新增 `adapt()` 方法，允许用户在运行时主动切换或观察自动切换：

```python
engine = V9Orchestrator(lambda_estimate=0.1)  # Edge 模式
engine.step("query 1", tool=search)
engine.adapt(lambda_estimate=0.001)            # 换到 PC 模式
```

内部每 N 步（N=100）自动重评估一次模式：`self._mode = self._barrier.evaluate()["recommended_operator"]`。

## 3. 架构图

```
┌─────────────────────────────────────────────┐
│              V9Orchestrator                  │
│                                              │
│  step(content, tool) ──→ store memory        │
│       │                                      │
│       ├── Fractal mode  → 9D C_ij coupling   │
│       ├── ODE mode      → ont_self (default) │
│       ├── exp_log mode  → Palace aggregate   │
│       └── PRNG mode     → skip               │
│                                              │
│  report() ──→ { ont_self, ..., barrier: {    │
│       lambda, mode, scenario, collapse_risk  │
│  }}                                           │
│                                              │
│  adapt(lam) ──→ re-evaluate operator         │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│   vinfty/barrier/             │
│                              │
│  table.py                    │
│    ├── OPERATOR_TABLE        │  5 算子参数表
│    ├── compute_l2_star()     │  L₂* = L₂,min − τ·ln(λτ/ΔL₁)
│    ├── compute_l1_star()     │  L₁* = L₁,inf + ΔL₁·exp(−(L₂*−L₂,min)/τ)
│    ├── compute_l_total()     │  L_total* = L₁* + λ·L₂*
│    └── compute_p_trans()     │  P_trans = exp(−L₁/σ₁ − L₂/σ₂)
│                              │
│  estimator.py                │
│    ├── LambdaEstimator       │  硬件指标 → λ
│    └── auto_detect()         │  自动检测 PC/Edge/MCU
│                              │
│  scheduler.py                │
│    ├── LambdaScheduler       │  评价 + 推荐
│    ├── CollapseDetector      │  寂静坍缩检测
│    └── recommend()           │  T7-compatible 接口
└──────────────────────────────┘
```

## 4. 与 vinfty 现有模块的关系

```
vinfty/
├── __init__.py          # 导出 V9Orchestrator (不变)
├── core.py              # V9Orchestrator + barrier 集成 (改 ~40 行)
├── barrier/             # 新增
│   ├── __init__.py
│   ├── table.py         # 核心公式 (无依赖)
│   ├── estimator.py     # 硬件检测 (platform + os 标准库)
│   └── scheduler.py     # 调度决策 (依赖 table + estimator)
└── adapters/
    ├── __init__.py
    └── langchain.py      (不变)
```

## 5. 验收标准

| # | 验收项 | 方法 |
|---|--------|------|
| 1 | `barrier/table.py` 5 算子参数与 SiliconLifeOS `lambda_scheduler.py` 一致 | `compute_operator_optimal(0.001, ODE) → L_total*=22.17` |
| 2 | `auto_detect()` 在 PC 上返回 ~0.001 | `pytest` |
| 3 | `V9Orchestrator(lambda_estimate=0.1)` 创建的 engine 自动为 exp_log 模式 | `engine._mode == "exp_log"` |
| 4 | `adapt(1e-5)` 后模式切换为 Fractal | `engine._mode == "Fractal"` |
| 5 | Fractal 模式 report() 含 `barrier.collapse_risk` | `assert "barrier" in report` |
| 6 | 回归：`lambda_estimate=None` 默认行为与现有 API 完全一致 | 原有 10 个测试全过 |
| 7 | 零外部依赖 | `pip install vinfty` 不自带 numpy/torch |
| 8 | CPU-only + log-only，不写文件 | CI 验证 |

## 6. 里程碑

| 阶段 | 交付 | 行数 |
|------|------|------|
| P0 | `barrier/table.py` 纯函数 + 单元测试 | ~80 |
| P1 | `barrier/estimator.py` + `auto_detect()` | ~50 |
| P2 | `barrier/scheduler.py` + `V9Orchestrator` 集成 | ~80 |
| P3 | `adapt()` + 动态切换 + 回归测试 | ~40 |
| 验收 | 8/8 标准通过 | — |

总计约 **250 行新增**，零额外依赖。现有测试全绿。

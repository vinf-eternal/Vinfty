# Vinfty — AI 一致性审计辅助引擎

**Vinfty 不是 AI 模型。它不训练、不拟合、不推理、不预测。**
它只做一件事：作为外挂式辅助组件，**审计 AI 系统内部的自洽性**。

```
pip install vinfty
```

---

## 定位

> Vinfty ≠ 另一个 LLM / 小样本分类库 / 微调框架
> Vinfty = **AI Consistency Auxiliary Auditor**

现有 AI 工具链（LangChain、LlamaIndex、EvidentlyAI、Prometheus）只追踪：
- 日志、延迟、吞吐量
- 准确率、召回率、损失曲线

Vinfty 补上唯一缺失的维度：**认知一致性**。

| 问题 | 传统工具 | Vinfty |
|------|---------|--------|
| 模型精度 99%？ | ✅ 是 | ❌ 不管 |
| 内部自洽性在恶化？ | ❌ 看不见 | ✅ ont_self 追踪 |
| 标注数据有没有矛盾簇？ | ❌ 人工抽检 | ✅ 自动审计 |
| 小样本训练会不会先天跑偏？ | ❌ 凭经验 | ✅ 辅助预警 |
| 线上模型精度还没跌，但内部已在坍缩？ | ❌ 不知道 | ✅ 寂静坍缩检测 |

---

## 模块

| 模块 | 用途 | 状态 |
|------|------|------|
| `vinfty.judge` | 数据集一致性审计辅助（ont_self 打分、矛盾簇定位、标注冲突报告） | ✅ 可用 |
| `vinfty.barrier` | 势垒经济学自适应观测深度调度（λ 控制审计精细度） | ✅ 可用 |
| `vinfty.core` | V9Orchestrator 认知观测内核（300 行，零依赖） | ✅ 稳定 |
| `vinfty.filter` | 小样本训练过程监控辅助（预留接口） | 📌 暂未实现 |
| `vinfty.trace` | 线上模型认知漂移预警辅助（预留接口） | 📌 暂未实现 |

---

## Quick Start：数据集审计

```python
from vinfty.judge import audit_dataset

texts = [
    "The cat sat on the mat.",
    "The dog slept on the rug.",
    "神经网络通过反向传播优化参数。",
    "Transformer 使用自注意力机制。",
    "Gradient descent minimizes the loss function.",
    "北京是中国的首都。",
    "Paris is the capital of France.",
    "The quick brown fox jumps over the lazy dog.",
]

report = audit_dataset(texts, lambda_estimate=0.001)
print(report["ont_self_mean"])         # 0.23
print(report["n_palaces"])             # 7
print(report["contradictions_top5"])   # 不一致样本对
print(report["silent_collapse_risk"])  # green / yellow / red
```

### 输出示例

```json
{
  "n_samples": 8,
  "ont_self_mean": 0.23,
  "c_ij_density": 0.11,
  "n_palaces": 7,
  "palace_distribution": {
    "P_hash_02": 2,
    "P_hash_15": 1,
    "P_hash_25": 1,
    "P_hash_42": 1,
    "P_hash_51": 1,
    "P_hash_57": 1,
    "P_hash_61": 1
  },
  "label_conflicts": [],
  "silent_collapse_risk": "green",
  "hmM_state": "s0",
  "barrier": {
    "lambda": 0.001,
    "mode": "ODE",
    "scenario": "pc_benchmark",
    "L_total_star": 22.17,
    "P_trans": 0.51
  }
}
```

### 带标注的数据集

```python
report = audit_dataset(
    texts,
    labels=["animal", "animal", "ML", "ML", "ML", "geo", "geo", "noise"],
    lambda_estimate=0.001,
)
print(report["label_conflicts"])
# → 如果同 label 映射到不同 Palace，说明标注边界有问题
```

---

## 势垒经济学：审计也有成本

`lambda_estimate` 控制审计深度——不是模型精度权衡，是 **观测开销-观测深度调度**。

| 部署场景 | λ | 审计策略 |
|----------|---|---------|
| PC/服务器 | 0.001 | 深度精细审计，完整 9D 耦合计算 |
| 数据中心 | 1e-5 | 高密度嵌套观测，批量扫描海量日志 |
| 边缘设备 | 0.1 | 轻量化聚合审计，保留核心异常检出 |
| MCU 嵌入式 | 100 | 极简统计式快速巡检 |

```python
report = audit_dataset(texts, lambda_estimate=0.1)  # 边缘端轻审计
```

---

## No LLM Required

Vinfty 是纯符号计算——不需要 API Key、GPU、模型权重。
所有认知指标来自**调用序列和记忆痕迹的结构**，而非 LLM 响应的内容。

---

## License

MIT

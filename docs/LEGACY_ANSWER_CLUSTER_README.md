# Answer-Cluster RM-Support

基于答案簇证据的奖励模型候选重排实验。最终版本冻结于 Git 标签 `answer-cluster-final-v1`，对应提交 `0e8d5dbe9ce07e52668ae52ccb79f419d8cdb202`。

## 1. 项目结论

本项目研究：当 pointwise 奖励模型已经能对单条解答打分时，能否利用同一道题多个候选之间的结构化信息进一步提高最终答案选择准确率。

最终有效方法不是高维隐状态校正，而是低维、可解释的答案簇证据：

1. 从每条候选中解析并规范化最终答案；
2. 将最终答案相同的候选合并成答案簇；
3. 综合答案簇支持数量、解答多样性和基础 RM 分数统计；
4. 训练轻量答案簇排序器；
5. 仅在 Pilot 选择的证据阈值满足时替换原始 RM Top-1；
6. 在选定答案簇内保留基础 RM 分数最高的候选。

在 GSM8K ID、MATH ID 和 SVAMP OOD 三个冻结测试集上，最终方法将宏平均 Top-1 从 **0.808097** 提升至 **0.830445**，将 Pair-Macro 从 **0.830883** 提升至 **0.909742**。

## 2. 冻结版本

- Git 标签：`answer-cluster-final-v1`
- Git 提交：`0e8d5dbe9ce07e52668ae52ccb79f419d8cdb202`
- 基础奖励模型：`Skywork-Reward-V2-Qwen3-1.7B`
- 最终配置：`configs/answer_cluster_final.json`
- 测试标签参与训练或调参：否

## 3. 数据协议

| 数据集 | 角色 | 问题数 | 候选数 | 生成模型 |
|---|---:|---:|---:|---|
| GSM8K | 训练 | 1,531 | 24,221 | Qwen2-1.5B |
| GSM8K | Pilot | 189 | 2,985 | Qwen2-1.5B |
| GSM8K | ID 测试 | 1,142 | 17,902 | Qwen2-1.5B |
| MATH | 训练 | 1,276 | 16,229 | Qwen2-7B |
| MATH | Pilot | 158 | 2,040 | Qwen2-7B |
| MATH | ID 测试 | 340 | 4,167 | Qwen2-7B |
| SVAMP | OOD 测试 | 883 | 13,913 | Qwen2-1.5B |

训练集用于拟合答案簇模型，Pilot 用于选择正则化、残差强度和切换阈值。三个测试集仅在配置冻结后用于评估。

## 4. 最终方法

### 4.1 答案簇

候选回答通过最终答案解析器归一化。答案相同的候选合并为一个簇。解析覆盖率接近 100%，簇标签纯度超过 99.9%。少量混合标签簇保留在评测中，但不能构成明确正负簇的问题不会用于训练正负对。

### 4.2 最终特征

- 答案簇候选数量；
- 答案簇占该题全部候选的比例；
- 簇内不同解答文本数量；
- 相对簇内最大奖励分；
- 相对簇内平均奖励分；
- 相对簇内 log-mean-exp 奖励；
- 相对簇内奖励标准差；
- 原始 RM Top-1 所在答案簇指示量。

最终模型不使用生成模型 token NLL 或生成模型隐状态，因为二者没有带来稳定的全量测试增益。

### 4.3 冻结参数

| 领域 | 正则化 | Beta | 切换阈值 |
|---|---:|---:|---:|
| GSM8K / SVAMP | 0.1 | 2.0 | 0.25 |
| MATH | 0.001 | 2.0 | 0.25 |

## 5. 最终结果

### 5.1 各数据集

| 数据集 | Raw Top-1 | Final Top-1 | ΔTop-1 | Raw Pair | Final Pair | ΔPair | Damage |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K ID | 0.830998 | 0.842382 | +0.011384 | 0.857464 | 0.928776 | +0.071312 | 0.016860 |
| MATH ID | 0.723529 | 0.773529 | +0.050000 | 0.785060 | 0.864710 | +0.079650 | 0.036585 |
| SVAMP OOD | 0.869762 | 0.875425 | +0.005663 | 0.850126 | 0.935741 | +0.085615 | 0.011719 |
| **宏平均** | **0.808097** | **0.830445** | **+0.022349** | **0.830883** | **0.909742** | **+0.078859** | **0.021721** |

### 5.2 配对 Bootstrap

以问题为单位执行 5,000 次配对 Bootstrap：

| 指标 | 点估计 | 95% 置信区间 |
|---|---:|---:|
| ΔTop-1 | +0.022349 | [+0.009487, +0.034607] |
| ΔPair-Macro | +0.078859 | [+0.069251, +0.088275] |
| Damage | 0.021721 | [0.013540, 0.031076] |
| ΔBest@4 | +0.057225 | [+0.050220, +0.064261] |
| ΔBest@8 | +0.046716 | [+0.036738, +0.056946] |

所有主要收益指标的置信区间都严格大于零。

### 5.3 候选预算

`Pass@k` 是随机 k 候选中至少存在一个正确答案的理论概率；算法不能改变它。`Best@k` 是随机 k 候选子集中，排序方法所选最高分候选的期望正确率。

| 方法 | Best@4 | Pass@4 | Best@8 | Pass@8 |
|---|---:|---:|---:|---:|
| Raw RM | 0.699786 | 0.820271 | 0.768492 | 0.934893 |
| Final RM-Support | 0.757011 | 0.820271 | 0.815208 | 0.934893 |
| 提升 | +0.057225 | — | +0.046716 | — |

最终方法关闭了约 47.50% 的 Best@4 oracle gap 和约 28.08% 的 Best@8 oracle gap。

## 6. 核心消融

| 方法 | Macro Top-1 | ΔTop-1 | Macro Pair | ΔPair | Damage |
|---|---:|---:|---:|---:|---:|
| Raw RM | 0.808097 | 0 | 0.830883 | 0 | 0 |
| Reward-weighted consensus | 0.822507 | +0.014411 | 0.906913 | +0.076030 | 0.045037 |
| **RM-Support** | **0.830445** | **+0.022349** | **0.909742** | **+0.078859** | **0.021721** |
| RM-Support + NLL | 0.827967 | +0.019871 | 0.910096 | +0.079212 | 0.024000 |
| RM-Support + NLL + hidden | 0.827279 | +0.019182 | 0.908444 | +0.077560 | 0.029116 |

生成模型 NLL 只带来极小的 Pair 变化，却降低 Top-1 并增加 Damage；生成模型隐状态没有稳定增益。因此最终采用更简单的 RM-Support。

## 7. 与轨迹校正比较

| 方法 | Macro Top-1 | Macro Pair | Damage |
|---|---:|---:|---:|
| Raw RM | 0.808097 | 0.830883 | 0 |
| 安全轨迹校正 | 0.824277 | 0.849749 | 0.012086 |
| **答案簇 RM-Support** | **0.830445** | **0.909742** | **0.021721** |

轨迹校正具有更低 Damage，但答案簇方法的 Top-1 和 Pair-Macro 更强。

## 8. 外部奖励模型审计

四个未参与答案簇训练的 Skywork 奖励模型对 151 个发生选择变化的问题进行评分：

| 指标 | 结果 |
|---|---:|
| 多数偏好新选择 | 0.1987 |
| 多数偏好原始选择 | 0.5497 |
| 多数平票 | 0.2517 |
| 与真实标签变化对齐 | 0.5097 |
| 对真实修复偏好新答案 | 0.2609 |
| 对真实破坏偏好原答案 | 0.6176 |

1.7B 冻结分数复现达到 Pearson 0.99988、Pair 偏好一致率 97.35%，因此评分实现不存在足以解释上述结果的系统性错误。

外部 RM 结果不是最终方法的正面质量证据。它说明答案簇群体证据与 pointwise RM 的孤立回答评价之间存在明显错位；同属 Skywork 系列也可能造成对基础 RM 原选择的共同偏好。

## 9. 局限性

1. 最终测试只包括 GSM8K、MATH 和 SVAMP；
2. 候选主要来自 Qwen2-1.5B 和 Qwen2-7B；
3. SVAMP 只代表简单算术 OOD；
4. 外部裁判都属于 Skywork 系列；
5. 方法依赖可解析的最终答案，开放式任务需要新的聚类方式；
6. 数学等价答案归一化仍可能存在极少量符号误差。

## 10. 目录与结果来源

关键配置和结果：

```text
configs/answer_cluster_final.json
configs/environment_frozen.json
data/manifests/answer_cluster_generator_full_v1.json
data/manifests/answer_cluster_rm_support_bootstrap_v1.json
data/manifests/answer_cluster_holdout_rewards_v1.json
data/manifests/holdout_reward_scoring_reproduction.json
```

主要脚本：

```text
scripts/eval_answer_cluster_rm_support_full.py
scripts/eval_answer_cluster_generator_full.py
scripts/bootstrap_answer_cluster_final.py
scripts/eval_answer_cluster_holdout_rewards.py
scripts/audit_holdout_reward_scoring.py
```

## 11. 最小复现要求

Git Bundle 不包含模型、大型缓存和 `data/processed/`。复现最终 RM-Support 至少需要：

### 数据

```text
data/processed/prototype_v2/gsm_train.jsonl
data/processed/prototype_v2/gsm_pilot_validation.jsonl
data/processed/prototype_v2/gsm_id_test_mixed.jsonl
data/processed/prototype_v2/math_train.jsonl
data/processed/prototype_v2/math_pilot_validation.jsonl
data/processed/prototype_v2/math_id_test_mixed.jsonl
data/processed/prototype_v2/svamp_ood_mixed.jsonl
```

### 基础 RM 分数

```text
data/cache/trajectory_features_v1/
Skywork-Reward-V2-Qwen3-1.7B/layer_28/
```

至少保留上述目录中七个对应数据前缀的 `scores_f32.npy` 文件。

### 模型

```text
models/reward/Skywork-Reward-V2-Qwen3-1.7B
```

如果已经保存基础 RM 分数，只复现答案簇训练和评测时不需要再次加载模型。

## 12. 推荐复现顺序

```bash
python -u scripts/eval_answer_cluster_rm_support_full.py
python -u scripts/eval_answer_cluster_generator_full.py
python -u scripts/bootstrap_answer_cluster_final.py
```

外部奖励模型诊断需要四个额外模型：

```bash
python -u scripts/eval_answer_cluster_holdout_rewards.py
```

复现成功时应得到近似结果：

```text
Macro Top-1: 0.808097 -> 0.830445
Macro Pair:  0.830883 -> 0.909742
Damage:      0.021721
```

BF16 推理可能造成极小末位差异，但不应改变主要结论。

## 13. 防止数据泄漏

- 不得使用三个测试集选择正则化、Beta 或阈值；
- 最终参数只能由训练集和 Pilot 决定；
- Holdout RM 结果不得反向用于修改 `final_v1`；
- 新探索应创建新版本，不能覆盖冻结结果。

## 14. 恢复 Git Bundle

```bash
git clone rm_traj_answer_cluster_final_v1.bundle rm_traj_project
cd rm_traj_project
git checkout answer-cluster-final-v1
```

完整环境版本和关键文件 SHA256 记录在 `configs/environment_frozen.json`。

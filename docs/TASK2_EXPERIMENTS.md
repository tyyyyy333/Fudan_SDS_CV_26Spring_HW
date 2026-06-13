# Task 2 数据、训练与评价记录

## 数据来源与格式

- 原始数据：`huiwon/calvin_task_ABC_D`
- 原始页面：[https://huggingface.co/datasets/huiwon/calvin_task_ABC_D](https://huggingface.co/datasets/huiwon/calvin_task_ABC_D)
- 本项目复现副本：[https://huggingface.co/datasets/WhySoTy/hw3_calvin_40g](https://huggingface.co/datasets/WhySoTy/hw3_calvin_40g)
- shard：`0_4=A`、`1_4=B`、`2_4=C`、`3_4=D`

远端压缩文件约 7.81 GB；差异来自 Hub 统计、
缓存、元数据和文件系统口径。原始 LeRobot v3 数据用 AV1 视频保存两路 256x256
RGB 相机，以 Parquet 保存状态、动作、任务和索引。训练随机访问帧时 AV1 解码
成为瓶颈，因此按完整 episode 转为 LeRobot `image` 特征。

```bash
conda activate hw3t2
python scripts/build_calvin_fast_subset.py \
  --target-frames-total 370000 \
  --store image \
  --seed 23300200022
```

主种子为 `23300200022`，A/B/C/D 实际种子依次为 `23300200022` 至
`23300200025`。只选择完整 episode，不截断、不生成新样本：

| 环境 | Episodes | Frames | 用途               |
| ---- | -------: | -----: | ------------------ |
| A    |     1541 |  92409 | ABC 训练           |
| B    |     1532 |  92581 | B-only 与 ABC 训练 |
| C    |     1536 |  92688 | ABC 训练           |
| D    |     1538 |  92274 | 仅 zero-shot 评价  |

最终解码数据为 369952 帧、39.958 GB。D 从未参与梯度更新。

## 模型与控制变量限制

两个 checkpoint 使用相同 ACT 架构：ResNet-18、hidden dim 512、8 heads、
4 层 encoder、1 层 decoder、latent dim 32、KL weight 10、chunk size 10、
batch 256、seed 1000。

| 配置   | 数据  | Steps | LR schedule |
| ------ | ----- | ----: | ----------- |
| B-only | B     | 10000 | 500 warmup + cosine to `1e-5` |
| A+B+C  | A+B+C | 10000 | 500 warmup + cosine to `1e-5` |

两组均执行 10000 次参数更新、每步 256 个样本，共处理 256 万训练样本；优化器、
500-step warmup、cosine decay、梯度裁剪、图像预处理、随机种子和保存频率完全相同。公平性由
`scripts/audit_task2_fairness.py` 直接比较两个 checkpoint 的 `train_config.json`。
唯一允许的差异是 B-only 使用环境 B，而 A+B+C 使用环境 A、B、C。

早期实验采用 B-only 5k/fixed 与 A+B+C 30k/cosine，完整 D Action L1
分别为 0.508989 和 0.419402。该结果混合了数据范围、训练步数和调度器，不再作为
正式结论；指标与图保存在
`outputs/task2/experiments/unfair_b5k_vs_abc30k/`，用于说明为何必须重做控制实验。

## 完整 D Zero-shot 动作误差

所有数字来自
`outputs/task2/zero_shot_d_action_error_full/metrics.json`，属于 92274 样本的
教师强制离线动作误差，不是 simulator rollout success rate。

| 配置              | Action L1 | Total loss | KLD |
| ----------------- | --------: | ---------: | --: |
| B-only 10k/cosine |  0.509418 |   0.509554 | 0.00001358 |
| A+B+C 10k/cosine  | **0.432757** | **0.432869** | **0.00001120** |

严格同参下，A+B+C 的 Action L1 和总损失均降低 15.05%。

## Action Chunking 与视觉偏移

受控分析使用 B/D 各 8192 个样本、chunk size 10，并保持机器人状态和标签不变，
只扰动输入图像。主要结果：

- clean D L1：`0.537354 -> 0.462663`
- arm cosine：`0.552400 -> 0.649600`
- gripper sign accuracy：`90.729% -> 92.854%`
- tail/head ratio：`1.020170 -> 1.018037`
- prediction/GT variation ratio：`0.506615 -> 0.405908`
- 外观偏移平均绝对 L1 优势：11.515%
- 模糊/噪声平均绝对 L1 优势：10.124%
- 相机平移/中心遮挡平均绝对 L1 优势：4.795%
- 相机缺失平均绝对 L1 优势：4.852%

误差在十个 chunk 位置上没有递归爆炸，说明联合解码提供了短时一致性。但当前配置
连续执行 10 步且没有 temporal ensemble；错误视觉条件会共同污染整个 chunk，
期间无法利用新观测纠正。Action Chunking 能降低 chunk 内递归误差，不会自动获得
相机几何不变性。腕部相机缺失造成最大绝对误差；A+B+C 的误差仍较低，但相对自身
clean D 退化 63.75%，说明近场视觉既关键，也是主要 domain-shift 通道。

官方 CALVIN 仿真和 validation 资源已因空间不足删除，因此没有闭环 success rate。
恢复 simulator、scene、task oracle 和 rollout evaluator 后才能运行
`scripts/entrypoints/task2/evaluate_d_rollout.sh`。

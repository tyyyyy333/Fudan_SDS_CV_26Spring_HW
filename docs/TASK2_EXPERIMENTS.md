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
batch 512、seed 1000。

| 配置   | 数据  | Steps | LR schedule        |
| ------ | ----- | ----: | ------------------ |
| B-only | B     |  5000 | fixed              |
| A+B+C  | A+B+C | 30000 | 1k warmup + cosine |

最终比较并非严格的数据单变量实验，因为步数和调度器不同。`17.60%` 只能解释为
两个最终配置的端到端差异，不能全部归因于环境多样性。A+B+C 的 10k/20k/30k
D 抽样 Action L1 为 0.42098/0.40852/0.40784，后段已接近饱和，但仍不能替代
缺失的 B-only 30k 控制组。

## 完整 D Zero-shot 动作误差

所有数字来自
`outputs/task2/zero_shot_d_action_error_full/metrics.json`，属于 92274 样本的
教师强制离线动作误差，不是 simulator rollout success rate。

| 配置             |          Action L1 |         Total loss |                  KLD |
| ---------------- | -----------------: | -----------------: | -------------------: |
| B-only 5k/fixed  |           0.508989 |           0.510606 |           0.00016165 |
| A+B+C 30k/cosine | **0.419402** | **0.419420** | **0.00000176** |

最终配置 Action L1 差异为 17.6010%，总损失差异为 17.8585%。

## Action Chunking 与视觉偏移

受控分析使用 B/D 各 8192 个样本、chunk size 10，并保持机器人状态和标签不变，
只扰动输入图像。主要结果：

- clean D L1：`0.544000 -> 0.446679`
- arm cosine：`0.535381 -> 0.652701`
- gripper sign accuracy：`90.758% -> 92.445%`
- tail/head ratio：`0.999824 -> 1.017028`
- prediction/GT variation ratio：`0.425275 -> 0.584879`
- 外观偏移平均绝对 L1 优势：14.853%
- 模糊/噪声平均绝对 L1 优势：17.725%
- 相机平移/中心遮挡平均绝对 L1 优势：1.553%
- 相机缺失平均绝对 L1 优势：7.993%

误差在十个 chunk 位置上没有递归爆炸，说明联合解码提供了短时一致性。但当前配置
连续执行 10 步且没有 temporal ensemble；错误视觉条件会共同污染整个 chunk，
期间无法利用新观测纠正。Action Chunking 能降低 chunk 内递归误差，不会自动获得
相机几何不变性。静态相机缺失时 A+B+C 的 tail/head ratio 增至 1.096；腕部相机
缺失造成最大绝对误差，说明近场视觉既关键，也是主要 domain-shift 通道。

官方 CALVIN 仿真和 validation 资源已因空间不足删除，因此没有闭环 success rate。
恢复 simulator、scene、task oracle 和 rollout evaluator 后才能运行
`scripts/entrypoints/task2/evaluate_d_rollout.sh`。

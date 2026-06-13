# 运行、排障与打包

## 正式入口

| 入口                                                   | 环境      | 用途            |
| ------------------------------------------------------ | --------- | --------------- |
| `scripts/entrypoints/task1/train_background.sh`      | `hw3t1` | kitchen 2DGS    |
| `scripts/entrypoints/task1/train_object_a.sh`        | `hw3t1` | A COLMAP + 2DGS |
| `scripts/entrypoints/task1/train_object_b.sh`        | `hw3t1` | B pure SDS      |
| `scripts/entrypoints/task1/train_object_c_coarse.sh` | `hw3t1` | C coarse        |
| `scripts/entrypoints/task1/train_object_c_fine.sh`   | `hw3t1` | C DMTet fine    |
| `scripts/entrypoints/task1/preview_fusion.sh`        | `hw3t1` | 12 帧预览       |
| `scripts/entrypoints/task1/finish_all.sh`            | `hw3t1` | 360 帧正式结果  |
| `scripts/entrypoints/task2/train_b_only.sh`          | `hw3t2` | B-only fair 10k |
| `scripts/entrypoints/task2/train_abc.sh`             | `hw3t2` | ABC fair 10k    |
| `scripts/entrypoints/task2/evaluate_fair_10k.sh`     | `hw3t2` | 公平审计与 D 评价 |
| `scripts/evaluate_task2_d_offline.py`                | `hw3t2` | 完整 D 动作误差 |
| `scripts/analyze_task2_action_chunking.py`           | `hw3t2` | chunk 鲁棒性    |
| `scripts/audit_report_metrics.py`                    | `hw3t1` | 报告数字审计    |

`scripts/archive/` 只保存报告引用的历史失败实现，不用于生成正式结果。

## 常见问题

### B negative prompt 类型错误

```bash
NEGATIVE_PROMPT='' bash scripts/entrypoints/task1/train_object_b.sh
```

脚本内部必须传入真正的空字符串，不能让 OmegaConf 得到 `None`。

### A 白色射线

正式方案保留原 rotation/scale/opacity/SH，只使用前景支持和：

```text
--object-a-footprint-mode original
--object-a-max-scale-ratio 0.12
--object-a-max-anisotropy 25
```

不要恢复统一圆盘、统一 opacity、清零 SH 或 sprite 路线。

### C 侧面/背面退化

不要启用 `--object-c-reproject-image`。正式融合只读 OBJ UV 和 `albedo.png`。

### MP4 无法播放

```bash
conda run -n hw3t1 python scripts/reencode_mp4_h264.py VIDEO.mp4 --in-place
conda run -n hw3t1 python scripts/inspect_mp4.py VIDEO.mp4
```

正式输出应为 H.264、`yuv420p` 和 faststart。

### PLY 怎么看

2DGS PLY 不能按普通点云判断最终外观。使用官方 rasterizer 或：

```bash
conda run -n hw3t1 python scripts/render_2dgs_camera_views.py --help
```

### Task2 success rate

离线图像/动作数据只能计算教师强制动作误差。闭环成功率还需要 CALVIN simulator、
validation scene、task oracle 和 rollout evaluator，不能由 Action L1 推算。

## 报告构建与审计

```bash
conda run -n hw3t1 python scripts/audit_report_metrics.py
conda run -n hw3t1 python scripts/build_hw3_report_assets.py
cd docs/report_cvpr
conda run -n hw3t1 tectonic hw3_report.tex --keep-logs
```

## 网盘证据包

在仓库根目录执行：

```bash
mkdir -p release/HW3_results/{report,task1,task2}

cp docs/report_cvpr/hw3_report.pdf release/HW3_results/report/
cp outputs/task1/final/scene/fused_scene_360.mp4 \
  outputs/task1/final/scene/fused_scene_360_contact.jpg \
  release/HW3_results/task1/
cp outputs/task1/final/quality/object_a_turntable_dense/turntable_360.mp4 \
  release/HW3_results/task1/object_a_turntable_360.mp4
cp outputs/task1/final/quality/object_b_turntable/turntable.mp4 \
  release/HW3_results/task1/object_b_turntable.mp4
cp outputs/task1/final/quality/object_c_turntable/turntable.mp4 \
  release/HW3_results/task1/object_c_turntable.mp4
cp outputs/task1/final/quality/environment_kitchen/contact.jpg \
  release/HW3_results/task1/environment_kitchen_contact.jpg

tar -I 'zstd -19' -cf release/HW3_results/task1/failure_evidence.tar.zst \
  outputs/task1/experiments/failure_evidence
tar -I 'zstd -19' -cf release/HW3_results/task2/zero_shot_d_action_error_full.tar.zst \
  outputs/task2/zero_shot_d_action_error_full
tar -I 'zstd -19' -cf release/HW3_results/task2/action_chunking_robustness.tar.zst \
  outputs/task2/action_chunking_robustness

find release/HW3_results -type f -print0 \
  | sort -z | xargs -0 sha256sum \
  > release/HW3_results/checksums.sha256
```

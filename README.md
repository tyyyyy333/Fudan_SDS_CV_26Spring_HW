# HW2 复现与提交说明

## 项目内容

本仓库对应 `HW2_计算机视觉.pdf` 的三项任务

| 任务   | 内容                                        | 当前主结果                                   |
| ------ | ------------------------------------------- | -------------------------------------------- |
| Task 1 | Oxford-IIIT Pet 细粒度分类                  | `swin_tiny` `test_acc = 93.21%`          |
| Task 2 | VisDrone 检测训练、视频跟踪、遮挡与越线分析 | `YOLOv8m + BoT-SORT` `line_count = 3`    |
| Task 3 | Oxford-IIIT Pet 三类语义分割                | `U-Net + CE + Dice` `test_mIoU = 0.7901` |

## 工程结构

| 路径               | 作用                           |
| ------------------ | ------------------------------ |
| `configs/`       | 三个任务的主配置文件           |
| `scripts/`       | 训练、调参、跟踪、图表生成脚本 |
| `src/hw2_cv/`    | 数据、模型、训练与分析实现     |
| `report_assets/` | 报告中使用的图表与拼图         |
| `outputs/`       | 三个任务的实验结果与导出文件   |
| `runs/`          | Ultralytics 检测训练目录       |

## 环境

建议直接使用已有 `cv` 环境

```bash
conda create -n cv python=3.10
python -m pip install -r requirements.txt
```

## 数据与大文件放置位置

网盘下载后的内容需要按下表放回项目目录，否则默认配置无法直接运行

| 网盘内容                          | 放置目录                         | 用途                          |
| --------------------------------- | -------------------------------- | ----------------------------- |
| Oxford-IIIT Pet 分类 parquet 数据 | `data/oxford-iiit-pet/`        | Task 1                        |
| Oxford-IIIT Pet 分割 parquet 数据 | `data/oxford_iiit_pet_hf_seg/` | Task 3                        |
| VisDrone 原始数据集               | `data/VisDrone2019-DET/`       | Task 2 检测训练               |
| 演示视频 `demo.mp4`             | `data/videos/demo.mp4`         | Task 2 跟踪与计数             |
| Task 1 实验结果压缩包             | `outputs/task1/`               | Task 1 结果复核与报告         |
| Task 2 实验结果压缩包             | `outputs/task2/`               | Task 2 视频、事件帧、分析结果 |
| Task 3 实验结果压缩包             | `outputs/task3/`               | Task 3 指标与预测可视化       |
| Ultralytics 检测训练目录          | `runs/`                        | Task 2 最佳权重来源           |

如果网盘中已经提供完整 `outputs/` 和 `runs/`，直接解压到项目根目录即可，形成如下路径

| 目标路径                  | 说明                |
| ------------------------- | ------------------- |
| `HW2/outputs/task1/...` | 分类实验结果        |
| `HW2/outputs/task2/...` | 检测与跟踪结果      |
| `HW2/outputs/task3/...` | 分割实验结果        |
| `HW2/runs/detect/...`   | YOLO 训练权重与曲线 |

## 复现命令

### Task 1

```bash
python scripts/run_task1_train.py --config configs/task1_baseline.yaml
python scripts/run_task1_sweep.py --config configs/task1_baseline.yaml
python scripts/run_task1_tune.py --config configs/task1_baseline.yaml
python scripts/run_task1_train.py --config outputs/task1/tuning/best_config.yaml
```

### Task 2

```bash
python scripts/prepare_visdrone.py --raw-root data/VisDrone2019-DET --output-root data/visdrone_yolo --data-yaml configs/visdrone_data.yaml
python scripts/run_task2_train.py --config configs/task2_visdrone.yaml
python scripts/run_task2_track.py --config configs/task2_visdrone.yaml
```

### Task 3

```bash
python scripts/run_task3_sweep.py --config configs/task3_unet.yaml
```

### 报告图表

```bash
python scripts/generate_report_assets.py
```

## 部分输出

| 任务   | 文件                                                                                     |
| ------ | ---------------------------------------------------------------------------------------- |
| Task 1 | `outputs/task1/*/summary.json` `outputs/task1/tuning/`                               |
| Task 2 | `outputs/task2/high_score/summary.json` `outputs/task2/high_score_track/tracked.mp4` |
| Task 3 | `outputs/task3/*/summary.json` `outputs/task3/ce_dice/prediction_exports/`           |
| 报告   | `REPORT.md` `report_assets/*.png`                                                    |

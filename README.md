# HW3: 2DGS Scene Construction and CALVIN ACT

- GitHub: [https://github.com/tyyyyy333/Fudan_SDS_CV_26Spring_HW/tree/HW3](https://github.com/tyyyyy333/Fudan_SDS_CV_26Spring_HW/tree/HW3)
- 数据与模型权重: [https://huggingface.co/datasets/WhySoTy/hw3_calvin_40g](https://huggingface.co/datasets/WhySoTy/hw3_calvin_40g)
- 最终报告: `docs/report_cvpr/hw3_report.pdf`

## Environments

| 任务   | Conda 环境 | Python | 主要依赖                                                 |
| ------ | ---------- | -----: | -------------------------------------------------------- |
| Task 1 | `hw3t1`  |   3.10 | PyTorch 2.3.1+cu121, COLMAP, 2DGS, threestudio, Magic123 |
| Task 2 | `hw3t2`  |   3.12 | PyTorch 2.5.1+cu121, LeRobot, ACT                        |

可移植环境文件为 `environment-hw3t1.yml` 和 `environment-hw3t2.yml`。

```bash
git clone --branch HW3 https://github.com/tyyyyy333/Fudan_SDS_CV_26Spring_HW.git
cd Fudan_SDS_CV_26Spring_HW/HW3
conda env create -f environment-hw3t1.yml
conda env create -f environment-hw3t2.yml
bash scripts/entrypoints/setup/bootstrap_external.sh
bash scripts/entrypoints/setup/check_environment.sh
```

外部仓库固定到实验所用 commit，并自动应用 `patches/` 中的本地修复。

## Directory Layout

```text
data/task1/object_a/             A 的原视频、469-view 正式数据和六张补充照片
outputs/task1/final/             Task 1 正式模型、质量图和 360 帧视频
outputs/task1/experiments/       有分析价值的失败对照
outputs/task2/                   ACT checkpoint、D zero-shot 和 chunking 分析
scripts/entrypoints/task1/       Task 1 shell 入口
scripts/entrypoints/task2/       Task 2 shell 入口
scripts/entrypoints/setup/       环境初始化与检查
scripts/tools/                   运维和导出 shell 工具
scripts/*.py                     数据、训练评价、渲染和报告实现
scripts/archive/                 已废弃实验和旧融合代码
docs/report_cvpr/                LaTeX 报告、图和 PDF
```

入口、排障和打包见 [docs/OPERATIONS.md](docs/OPERATIONS.md)，结果与发布边界见
[docs/ARTIFACTS.md](docs/ARTIFACTS.md)。

## Task 1

### Object A: COLMAP + 2DGS

正式数据含 469 个注册视角，保留视频帧 1-309、391-550。可靠相机方位覆盖约
281.6°，最大缺口约 78.4°。正式模型 30000 step、SH degree 3：

```bash
conda activate hw3t1
SOURCE="$PWD/data/task1/object_a/current" \
OUTPUT="$PWD/outputs/task1/final/objects/object_a/model" \
MAX_STEPS=30000 \
WHITE_BACKGROUND=1 \
bash scripts/entrypoints/task1/train_object_a.sh \
  > logs/object_a_469view_30k.log 2>&1
```

默认参数：

| 参数                 |                          默认值 |
| -------------------- | ------------------------------: |
| `SOURCE`           | `data/task1/object_a/current` |
| `MAX_STEPS`        |                           30000 |
| `WHITE_BACKGROUND` |                               1 |
| SH degree            |                               3 |
| 训练分辨率           |                原图 `960x533` |

正式指标：test L1 `0.008766`、test PSNR `30.4035 dB`；train L1
`0.005978`、train PSNR `30.1833 dB`。

同相机验证：

```bash
conda run -n hw3t1 python scripts/render_2dgs_camera_views.py \
  --source data/task1/object_a/current \
  --model outputs/task1/final/objects/object_a/model \
  --iteration 30000 \
  --output outputs/task1/final/quality/object_a_views \
  --num-views 16 --stride 5 --resolution 1 --white-background
```

A 的正式融合 PLY 通过前景支持获得：

```bash
conda run -n hw3t1 python scripts/filter_object_a_gaussians.py \
  --input outputs/task1/final/objects/object_a/model/point_cloud/iteration_30000/point_cloud.ply \
  --cameras outputs/task1/final/objects/object_a/model/cameras.json \
  --images data/task1/object_a/current/images \
  --output outputs/task1/final/objects/object_a/model/point_cloud/iteration_30000/point_cloud_supported.ply \
  --cache outputs/task1/final/objects/object_a/model/point_cloud/iteration_30000/support_cache.npz \
  --min-support-views 5 --min-support-ratio 0.70 --keep-all-supported
```

完整演变和 360° 缺失原因见
[docs/TASK1_EXPERIMENTS.md](docs/TASK1_EXPERIMENTS.md)。

### Object B: threestudio + Pure SDS

正式对象是汉堡，只使用文本、SD1.5 和 SDS：

```bash
conda activate hw3t1
RUN_ID=object_b_hamburger_official_v1 \
PROMPT='a delicious hamburger' \
NEGATIVE_PROMPT='' \
MAX_STEPS=10000 \
TRAIN_RESOLUTION=64 \
USE_PERP_NEG=false \
GUIDANCE_SCALE=100 \
SEED=0 \
WANDB_ENABLE=true \
WANDB_MODE=offline \
bash scripts/entrypoints/task1/train_object_b.sh \
  > logs/object_b_hamburger_official_v1.log 2>&1
```

从 checkpoint 导出 UV/MTL/纹理 OBJ：

```bash
WORKSPACE="$PWD/outputs/task1/final/objects/object_b/training" \
OUTPUT_DIR="$PWD/outputs/task1/final/objects/object_b/model" \
bash scripts/tools/export_task1_object_b_threestudio.sh
```

详细失败对照和 SDS 现象见
[docs/TASK1_EXPERIMENTS.md](docs/TASK1_EXPERIMENTS.md)。

### Object C: Magic123

正式结果为 v5，coarse 使用 SD+Zero123，fine 使用 DMTet：

```bash
conda activate hw3t1
RUN_ID=object_c_v5 \
bash scripts/entrypoints/task1/train_object_c_coarse.sh \
  > logs/object_c_v5_coarse.log 2>&1

RUN_ID=object_c_v5 COARSE_ID=object_c_v5 \
bash scripts/entrypoints/task1/train_object_c_fine.sh \
  > logs/object_c_v5_fine.log 2>&1
```

| 阶段   | Iters | SD/Zero123 lambda |    Guidance | 其他                |
| ------ | ----: | ----------------: | ----------: | ------------------- |
| coarse |  8000 |        `1 / 80` | `50 / 10` | seed 101, depth 0.1 |
| fine   |  5000 |   `1e-3 / 0.01` | `100 / 5` | DMTet               |

融合只读取 OBJ 的 UV 与 `albedo.png`，不把输入正面图覆盖到侧面或背面。实验历史见
[docs/TASK1_EXPERIMENTS.md](docs/TASK1_EXPERIMENTS.md)。

### Kitchen Background

```bash
conda activate hw3t1
MAX_STEPS=7000 bash scripts/entrypoints/task1/train_background.sh \
  > logs/task1_background_2dgs_7k.log 2>&1
```

输入为 Mip-NeRF 360 kitchen，正式模型含 279 个相机。

### Unified Fusion

正式路线把 A/B/C 全部转换为与背景相同的 2D Gaussian 参数，并追加到同一个
`GaussianModel`。每帧只调用一次官方 rasterizer。

- A：前景支持过滤；固定中心/半径/基；保留原 opacity、各向异性 scale、rotation
  和三阶 SH；SH 随刚体旋转换基。
- A：只删除长轴比例 `>0.12` 或各向异性 `>25` 的异常 footprint。
- B/C：按三角形面积采样 180000 点，面法线确定 rotation，面积密度确定 scale。
- C：只使用 OBJ UV 与 `albedo.png`。
- 相机、投影、alpha 合成和深度排序均与背景共享。

先渲染 12 帧：

```bash
conda activate hw3t1
bash scripts/entrypoints/task1/preview_fusion.sh
```

生成所有独立质量图和最终 360 帧：

```bash
bash scripts/entrypoints/task1/finish_all.sh \
  | tee logs/task1_finish_all.log
```

关键融合默认值：

| 参数                           |                 默认值 |
| ------------------------------ | ---------------------: |
| 帧数 / FPS                     |           `360 / 30` |
| scene scale                    |              `0.072` |
| B/C mesh samples               |            180000/物体 |
| B/C surfel scale / opacity     |        `1.45 / 0.92` |
| A footprint mode               |           `original` |
| A max scale ratio / anisotropy |          `0.12 / 25` |
| A/B/C ring radius              | `0.34 / 0.34 / 0.34` |
| A/B/C ring angle               |   `-0.8 / 0.8 / 1.8` |
| A/B/C scale                    | `0.95 / 1.45 / 1.30` |

结果：

```text
outputs/task1/final/scene/fused_scene_360.mp4
outputs/task1/final/scene/fused_scene_360_contact.jpg
```

## Task 2

数据来自 `huiwon/calvin_task_ABC_D`。远端压缩文件约 7.81 GB；本地解码为逐帧
LeRobot image 特征后为 39.958 GB、369952 帧。主抽样种子为 `23300200022`，
A/B/C/D 分别使用 `23300200022` 至 `23300200025`；ACT seed 为 `1000`。
详见 [docs/TASK2_EXPERIMENTS.md](docs/TASK2_EXPERIMENTS.md)。

训练入口：

```bash
conda activate hw3t2
bash scripts/entrypoints/task2/train_b_only.sh
bash scripts/entrypoints/task2/train_abc.sh
```

| 模型   | Batch | Steps | Chunk |       LR | Scheduler |
| ------ | ----: | ----: | ----: | -------: | --------- |
| B-only |   256 | 10000 |    10 | `1e-4` | 500 warmup + cosine |
| A+B+C  |   256 | 10000 |    10 | `1e-4` | 500 warmup + cosine |

完整 D zero-shot 动作误差：

```bash
conda activate hw3t2
python scripts/evaluate_task2_d_offline.py \
  --output-dir outputs/task2/zero_shot_d_action_error_full \
  --max-batches 0
```

| 模型        | D Action L1 | Total loss |
| ----------- | ----------: | ---------: |
| B-only 10k  |    0.509418 |   0.509554 |
| A+B+C 10k   | **0.432757** | **0.432869** |

严格同参下，A+B+C 相对 B-only 将完整 D Action L1 降低 15.05%。两组均为
10k steps、batch 256、500-step warmup、cosine decay 和 seed 1000，配置审计确认
唯一差异是训练数据范围。官方 CALVIN 仿真和 validation rollout 资源已因空间不足
删除；题目允许成功率或动作误差，本项目报告完整 D 动作误差。

Action chunking：

```bash
python scripts/analyze_task2_action_chunking.py \
  > logs/task2_action_chunking_robustness.log 2>&1
python scripts/plot_task2_action_chunking.py
```

## Quality Inspection

- A：`outputs/task1/final/quality/object_a_turntable_dense/turntable_360.mp4`
- B/C：各自 `object_*_turntable/contact.jpg` 与 `turntable.mp4`
- 环境：`outputs/task1/final/quality/environment_kitchen/contact.jpg`
- 融合：先看 `fused_scene_360_contact.jpg`，再看 MP4

服务器上检查 PLY 应使用官方 rasterizer；CloudCompare/MeshLab 只能粗看中心：

```bash
conda run -n hw3t1 python scripts/render_2dgs_camera_views.py --help
```

检查视频编码：

```bash
conda run -n hw3t1 python scripts/inspect_mp4.py \
  outputs/task1/final/scene/fused_scene_360.mp4
```

所有正式 MP4 应为 H.264、`yuv420p`、faststart。

## Report

```bash
conda run -n hw3t1 python scripts/audit_report_metrics.py
conda run -n hw3t1 python scripts/build_hw3_report_assets.py
cd docs/report_cvpr
conda run -n hw3t1 tectonic hw3_report.tex --keep-logs
```

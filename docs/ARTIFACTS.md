# 产物索引与发布清单

## 正式产物

| 路径                                               | 内容                            |
| -------------------------------------------------- | ------------------------------- |
| `outputs/task1/final/environment/kitchen_2dgs/`  | kitchen 2DGS 7k                 |
| `outputs/task1/final/objects/object_a/model/`    | A 469-view 30k 2DGS             |
| `outputs/task1/final/objects/object_b/model/`    | B 汉堡 OBJ/MTL/纹理             |
| `outputs/task1/final/objects/object_b/training/` | B checkpoint、CSV、TB、W&B      |
| `outputs/task1/final/objects/object_c/model/`    | C v5 OBJ/UV/albedo              |
| `outputs/task1/final/objects/object_c/training/` | C coarse/fine checkpoint 和日志 |
| `outputs/task1/final/quality/`                   | A/B/C/背景独立质量检查          |
| `outputs/task1/final/scene/fused_scene_360.mp4`  | 最终 360 帧融合视频             |
| `outputs/task2/runs/`                            | B-only 与 A+B+C checkpoint      |
| `outputs/task2/zero_shot_d_action_error_full/`   | 完整 D 指标                     |
| `outputs/task2/action_chunking_robustness/`      | chunk、视觉扰动和相机消融       |
| `docs/report_cvpr/hw3_report.pdf`                | 最终报告                        |

失败证据集中在 `outputs/task1/experiments/failure_evidence/`：

- `object_a_dense762/`：错误相机分支和重影。
- `object_b_kiwi/`：纯 SDS 语义/细节不足。
- `object_c_v6/`：背面黑化和 fine 退化。
- `fusion/`：旧二维代理、纹理投影和方向错误。

## GitHub

仓库：[https://github.com/tyyyyy333/Fudan_SDS_CV_26Spring_HW](https://github.com/tyyyyy333/Fudan_SDS_CV_26Spring_HW)
分支：`HW3`，项目目录：`HW3/`

GitHub 保存：

- README、环境 YAML、requirements、配置、`src/`、`scripts/`、`tests/`
- 精简后的 `docs/`、LaTeX、最终 PDF 和报告图
- 小型 JSON/CSV 指标、manifest 示例和校准参数

GitHub 不保存：

- `data/`、`external/`、`outputs/`、`logs/`、`weights/`
- checkpoint、PLY、OBJ 大文件、原始视频、逐帧图像、Conda 环境和缓存

## Hugging Face

数据与模型权重上传到：
[https://huggingface.co/datasets/WhySoTy/hw3_calvin_40g](https://huggingface.co/datasets/WhySoTy/hw3_calvin_40g)

至少应包含：

- CALVIN 40G 解码数据或其完整分片
- `subset_manifest.json` 和随机种子说明
- B-only 5k 与 A+B+C 30k checkpoint
- A 30k 2DGS PLY、B checkpoint/mesh、C coarse/fine checkpoint/mesh
- kitchen 2DGS 7k PLY

# HW2 扩展实验执行文档

## 1. 实验目标

这次扩展实验不重新训练 U-Net 主干，而是在 Task 3 已有 `CE + Dice` checkpoint 后增加一个非训练后处理模块：

`U-Net P0 -> image gradient -> anchor selection -> local propagation -> refined mask`

核心问题是：

- `border IoU` 是否能提升
- `mIoU` 是否不下降或小幅提升
- 可视化边界是否比原始 U-Net 输出更稳定

主实验只跑 Task 3。Task 1 和 Task 2 的扩展作为后续补充实验，不建议第一轮一起跑，避免变量太多。

## 2. 服务器环境

在服务器项目根目录执行：

```bash
cd /path/to/HW2
conda activate cv
python -m pip install -r requirements.txt
python -m pip install opencv-python-headless tqdm pyyaml albumentations
```

如果服务器没有 `cv` 环境：

```bash
conda create -n cv python=3.10 -y
conda activate cv
python -m pip install -r requirements.txt
```

## 3. 数据和 checkpoint 要求

确认以下路径存在：

```bash
ls data/oxford_iiit_pet_hf_seg
ls outputs/task3/ce_dice/best.pt
```

如果 `ce_dice` checkpoint 不在该路径，先查找：

```bash
find outputs/task3 -name best.pt
```

然后在运行命令里把 `--checkpoint` 改成实际路径。

如果只想把扩展实验文件同步到服务器，至少需要上传：

```bash
EXTENSION_EXPERIMENT.md
scripts/run_task3_anchor_refine.py
configs/task3_unet.yaml
src/hw2_cv/task3/
src/hw2_cv/cli.py
src/hw2_cv/runner.py
src/hw2_cv/utils.py
```

更稳妥的做法是直接同步整个项目目录，避免 `src` 内部 import 缺文件。

## 4. 先复核原始 Task 3 结果

如果服务器上还没有 `ce_dice` checkpoint，先跑原始 sweep：

```bash
python scripts/run_task3_sweep.py --config configs/task3_unet.yaml
```

预期已有结果大致为：

| setting | test mIoU | border IoU |
| --- | ---: | ---: |
| CE + Dice | 0.7901 | 0.5964 |

这个结果是扩展实验的 baseline。

## 5. 主实验：Gradient-Guided Anchor Refinement

默认运行：

```bash
python scripts/run_task3_anchor_refine.py \
  --config configs/task3_unet.yaml \
  --checkpoint outputs/task3/ce_dice/best.pt \
  --output-dir outputs/task3_anchor_refine/default
```

输出文件：

```bash
outputs/task3_anchor_refine/default/summary.json
outputs/task3_anchor_refine/default/prediction_exports/*_anchor_refine_compare.png
```

建议在服务器上用 `tmux` 或 `nohup` 留日志：

```bash
tmux new -s task3_anchor
mkdir -p outputs/task3_anchor_refine
python scripts/run_task3_anchor_refine.py \
  --config configs/task3_unet.yaml \
  --checkpoint outputs/task3/ce_dice/best.pt \
  --output-dir outputs/task3_anchor_refine/default \
  2>&1 | tee outputs/task3_anchor_refine/default.log
```

`summary.json` 中重点看：

```json
{
  "baseline": {
    "miou": "...",
    "classwise_iou": {
      "border": "..."
    }
  },
  "refined": {
    "miou": "...",
    "classwise_iou": {
      "border": "..."
    }
  },
  "delta": {
    "miou": "...",
    "classwise_iou": {
      "border": "..."
    }
  }
}
```

## 6. 推荐参数扫描

第一轮只扫轻量参数，不重新训练。

```bash
for conf in 0.85 0.90 0.95; do
  for grad in 0.08 0.12 0.16; do
    python scripts/run_task3_anchor_refine.py \
      --config configs/task3_unet.yaml \
      --checkpoint outputs/task3/ce_dice/best.pt \
      --output-dir outputs/task3_anchor_refine/conf_${conf}_grad_${grad} \
      --conf-threshold ${conf} \
      --grad-threshold ${grad}
  done
done
```

如果第一轮看到 `border IoU` 上升，再继续扫传播强度：

```bash
for strength in 0.35 0.55 0.75; do
  for fuse in 0.50 0.65 0.80; do
    python scripts/run_task3_anchor_refine.py \
      --config configs/task3_unet.yaml \
      --checkpoint outputs/task3/ce_dice/best.pt \
      --output-dir outputs/task3_anchor_refine/strength_${strength}_fuse_${fuse} \
      --diffusion-strength ${strength} \
      --fusion-weight ${fuse}
  done
done
```

## 7. 汇总结果

运行：

```bash
python - <<'PY'
import json
from pathlib import Path

rows = []
for path in sorted(Path("outputs/task3_anchor_refine").glob("*/summary.json")):
    s = json.loads(path.read_text())
    rows.append({
        "run": path.parent.name,
        "base_miou": s["baseline"]["miou"],
        "ref_miou": s["refined"]["miou"],
        "delta_miou": s["delta"]["miou"],
        "base_border": s["baseline"]["classwise_iou"]["border"],
        "ref_border": s["refined"]["classwise_iou"]["border"],
        "delta_border": s["delta"]["classwise_iou"]["border"],
    })

rows = sorted(rows, key=lambda x: (x["delta_border"], x["delta_miou"]), reverse=True)
print("run,base_miou,ref_miou,delta_miou,base_border,ref_border,delta_border")
for r in rows:
    print(",".join(str(r[k]) for k in r))
PY
```

## 8. 成功标准

建议用下面标准判断是否值得写进报告或课堂汇报：

| 等级 | 判定 |
| --- | --- |
| Strong | `border IoU` 提升至少 `+0.010`，且 `mIoU` 不下降 |
| Useful | `border IoU` 提升，`mIoU` 下降小于 `0.002`，可视化边界更稳定 |
| Negative | `border IoU` 和 `mIoU` 同时下降，说明传播过强或 anchor 选择不可靠 |

如果出现负结果，也可以讲成有价值的分析：

- 低梯度 anchor 可能不总是等价于语义稳定区域
- 局部传播可能过平滑，吃掉细边界
- 非训练后处理无法替代显式 boundary-aware training

## 9. 实测结果

我们在已有 `outputs/task3/ce_dice/best.pt` 上完成了后处理实验。结果如下：

| setting | mIoU | border IoU | delta mIoU | delta border IoU |
| --- | ---: | ---: | ---: | ---: |
| CE + Dice baseline | 0.790107 | 0.596366 | - | - |
| default anchor refine | 0.790567 | 0.596969 | +0.000460 | +0.000603 |
| best conf/grad sweep | 0.790591 | 0.597023 | +0.000484 | +0.000657 |
| best aggressive refine | 0.790797 | 0.597200 | +0.000690 | +0.000834 |

Best aggressive setting:

```bash
python scripts/run_task3_anchor_refine.py \
  --config configs/task3_unet.yaml \
  --checkpoint outputs/task3/ce_dice/best.pt \
  --output-dir outputs/task3_anchor_refine/strength_0.75_fuse_0.50 \
  --conf-threshold 0.95 \
  --grad-threshold 0.08 \
  --diffusion-strength 0.75 \
  --fusion-weight 0.50
```

结论：后处理稳定带来小幅正增益，但提升量不到 `+0.001` mIoU / border IoU，达不到 Strong 或 Useful 的标准。它更适合作为负结果分析：非训练式边界传播能修正少量局部边界，但不足以替代 boundary-aware loss、边界分支或更强的分割模型。

## 10. 预期写法

如果结果有效，可以在报告里写：

> We add a non-training gradient-guided anchor refinement module after the U-Net output. High-confidence and low-gradient interior pixels are selected as anchors, and their class probabilities are propagated to uncertain boundary regions. This targets the main residual error of Task 3, where the border class remains substantially weaker than foreground and background.

如果结果一般，可以写：

> The anchor refinement experiment improves interpretability but does not consistently improve mIoU, suggesting that post-hoc propagation alone is insufficient for boundary recovery. Future work should combine this idea with trainable boundary-aware losses or a lightweight boundary head.

## 11. 后续补充实验

Task 1 可以做 `segmentation-guided classification`：

- 输入 A：原图
- 输入 B：只保留 `pet`
- 输入 C：保留 `pet + border`
- 输入 D：原图和 mask 双分支

Task 2 可以做 `lost-recovery tracking`：

- 只在 `lost` 事件窗口附近低阈值补检
- 对短时 missing track 做线性插值
- 评价 `lost_count`、`line_count` 是否更稳定

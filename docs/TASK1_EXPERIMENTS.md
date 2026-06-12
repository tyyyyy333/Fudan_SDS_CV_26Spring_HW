# Task 1 实验记录

本文件合并 A/B/C 的版本演变、失败原因、正式参数和融合决策。最终定量结论以
`docs/report_cvpr/metric_audit.json` 和正式产物为准。

## Object A：陶瓷杯 COLMAP + 2DGS

| 项目 | 正式取值 |
|---|---|
| 输入 | `data/task1/object_a/source/IMG_2373.mp4` |
| 正式数据 | `data/task1/object_a/current` |
| 注册视角 | 469 |
| 保留帧段 | 1-309、391-550 |
| 拒绝帧段 | 310-390，相机轨迹漂移 |
| 训练 | 30000 step，SH degree 3，白背景 |
| Test | L1 0.00876596，PSNR 30.4035 dB |
| Train | L1 0.00597808，PSNR 30.1833 dB |
| 原始 / 融合 PLY | 66136 / 42800 Gaussians |
| 可靠方位覆盖 | 约 281.6°，最大缺口约 78.4° |

最初 44 视角模型只能稳定覆盖杯子一侧。密集抽帧后，前 550 帧形成主相机轨迹，
但强行加入更多帧会产生第二条漂移分支。白色釉面低纹理、移动高光、近似对称外形
及手部/背景特征共同破坏了 SfM 的静态刚体假设。相邻帧变化小不能保证长序列累计
位姿不漂移，因此最终删除 310-390，并拒绝用错误相机数量伪造完整 360°。

六张补充照片位于 `data/task1/object_a/supplement`。它们补充杯底和杯口，但焦距、
背景、遮挡、曝光和物体姿态与视频差异较大，未稳定注册到同一模型，只作定性证据。

### A 融合演变

| 路线 | 现象 | 结论 |
|---|---|---|
| 原始 2DGS 直接变换 | 未见角度出现白色长射线 | 极端各向异性 footprint 被侧视放大 |
| 2D sprite/正面贴图 | 无射线，但无真实背面和统一遮挡 | 不符合统一三维表达 |
| 重置 rotation/scale/opacity/SH | 射线消失，杯体和材质被破坏 | 破坏性补丁，已废除 |
| 最大连通分量 | 只剩 32887 点，薄杯壁被拆开 | 不适合分层 2DGS 表面 |
| 正式保真过滤 | 无白射线，保留视角相关外观 | 当前正式方案 |

正式过滤将 66136 个中心投影到 469 个前景视图，要求至少 5 次前景命中、命中率
不低于 0.70，再删除非有限值和 opacity 小于 0.02 的记录，得到 42800 个中心。
融合时使用固定 `fusion_calibration.json` 做刚体相似变换，只剔除归一化长轴大于
0.12 或各向异性大于 25 的 footprint。其余 rotation、两个切向 scale、opacity、
DC 和三阶 SH 均保留，SH 随刚体旋转换基。

该路线只能避免融合代码继续破坏资产，不能恢复 78.4° 未观测方位。A 的独立渲染
沿注册相机轨迹较好，而 kitchen 漫游进入覆盖缺口时仍可能看到孔洞、浮面或错误
高光。这是数据和位姿上限。

## Object B：threestudio 文本到 3D

正式路线只使用文本、threestudio DreamFusion 配置、Stable Diffusion 1.5 和纯
SDS，不使用 SDI、VSD、Zero123 或输入图像。

| 版本 | 对象/方法 | 结果 | 状态 |
|---|---|---|---|
| v1 | 恐龙、SDI | 依赖不兼容，且不符合最终约束 | 删除 |
| v2 | 交通锥、SDS+Perp-Neg | 尖端和薄底板破面 | 失败分析 |
| v3 | 蓝色几维鸟、纯 SDS | 单体轮廓可用，语义和细节不足 | 轻量失败证据 |
| v4 | 官方汉堡、纯 SDS 10k | 面包、蔬菜、肉层可辨 | 正式结果 |

交通锥的薄底板和尖端在 64x64 训练分辨率下容易被 density field 平滑；几维鸟的
长喙和翼面属于小尺度语义结构，SD1.5 SDS 不稳定。汉堡由厚、连续、分层体积组成，
更适合官方 DreamFusion 配置。

正式配置为 prompt `a delicious hamburger`、negative prompt 空字符串、
guidance 100、`weighting_strategy=sds`、`lambda_sds=1`、seed 0、10000 step。
训练耗时 38m48s。CSV、TensorBoard、W&B offline run 和每 200 步 RGB/normal/
opacity 验证图均保留。B 没有真实多视图 GT，因此不报告 PSNR；质量依据是训练
验证视图、法线/opacity 和 72 帧转盘。

## Object C：Magic123 单图到 3D

输入是手柄正面照片。核心困难是背面不可见，SD 容易复制正面语义形成 Janus 结构。

| 版本 | 关键变化 | 背面现象 | 状态 |
|---|---|---|---|
| v1 | Zero123 未正确激活 | 双头、黑背面 | 删除 |
| v2-v4 | 修复 Zero123、加入方向文本、简化 prompt | 双头减弱，背面仍像正面 | 中间实验 |
| v5 | Zero123 权重 80，SD/Zero123 guidance 50/10 | 单体，背面有凹陷 | 正式结果 |
| v6 | 强制 “plain white back” | 背面黑化、纹理退化 | 失败证据 |

v5 coarse 使用 8000 iter、SD/Zero123 lambda 1/80、guidance 50/10、seed 101、
depth 0.1；fine 使用 DMTet 5000 iter、lambda 0.001/0.01、guidance 100/5。
coarse/fine 训练分别为 29.2821/15.4947 分钟；从首次训练开始到最终 UV mesh
导出的日志跨度为 51m20s。

正式可视化以 `outputs/task1/final/quality/object_c_turntable/` 为唯一基准：
`00000` 为正面、`00045` 为斜视、`00060` 为凹陷背面。v5 消除了明显双头，但
背面结构和纹理仍是生成先验，不是观测真值。融合只读取 OBJ UV 与 `albedo.png`，
禁止用输入正面图重新覆盖侧面和背面。

## 统一 Gaussian 融合

- A 直接读取正式 2DGS PLY，并执行上述保真过滤和相似变换。
- B/C 从带纹理 OBJ 按三角形面积采样，每个对象 180000 个样本。
- 面法线确定 2D Gaussian 旋转，表面积和采样密度确定切向 footprint。
- B 使用 `texture_kd.jpg`；C 只使用 OBJ UV 与 `albedo.png`。
- A/B/C 追加到背景 `GaussianModel`，共享相机、投影、rasterizer、alpha 合成和
  深度排序。

正式融合是统一的可见表面表示，不是完整 PBR 材质迁移。B/C 的粗糙度、金属度、
镜面、透明和折射没有进入当前 shader；A 的陶瓷高光则保存在原始 SH 中。

最终场景：

```text
outputs/task1/final/scene/fused_scene_360.mp4
outputs/task1/final/scene/fused_scene_360_contact.jpg
```

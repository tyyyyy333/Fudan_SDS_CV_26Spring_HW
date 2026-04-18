# CV HW1 - Fashion-MNIST MLP

本项目实现了一个手写前向/反向传播的 MLP 分类器，支持预处理、模型结构、优化策略对比，以及可复现实验报告产物导出。

## 1. 文件说明

- `data_process.py`：数据下载、预处理模式（`baseline/flip/mask/flip_mask`）、DataLoader
- `layer.py`：Linear/ReLU/Sigmoid/Tanh/Dropout/CrossEntropy（含 backward）
- `model.py`：MLP 组网、参数管理、保存加载
- `optim.py`：SGD/Adam + Weight Decay + LR Decay
- `train.py`：单次训练、网格/随机搜索、实验汇总、详细报告生成
- `test.py`：独立测试、混淆矩阵、分组错例、错误表导出
- `visualization.py`：训练曲线、对比曲线、权重图、混淆矩阵图、错例图

## 2. 环境依赖

```bash
pip install numpy pandas matplotlib datasets tabulate
```

## 3. 如何开始

### 3.1 单次训练

```bash
python train.py \
  --exp_name baseline_run \
  --epochs 30 \
  --batch_size 128 \
  --optimizer sgd \
  --learning_rate 0.01 \
  --lr_decay 0.95 \
  --weight_decay 1e-4 \
  --hidden_sizes 256 \
  --activation relu \
  --dropout 0.2 \
  --preprocess_mode baseline \
  --save_plots
```

### 3.2 超参数搜索

```bash
python train.py \
  --exp_name exp_grid \
  --grid_search \
  --search_mode grid \
  --max_trials 20 \
  --epochs 30 \
  --save_plots \
  --save_csv
```

说明：

- 搜索阶段只用验证集选优；
- 最终只在最佳配置上评估一次测试集；
- 同时保留 baseline 对照并生成对比曲线。

### 3.3 独立测试与错例分析

注意：测试模型参数设置应与best模型参数设置相统一

```bash
python test.py \
  --exp_name exp_grid \
  --model_path outputs/exp_grid/best/best_model.npz \
  --hidden_sizes 128 \
  --activation tanh \
  --dropout 0 \
  --preprocess_mode baseline \
  --plot_cm \
  --plot_errors \
  --error_grouped \
  --top_confusions_k 5 \
  --save_error_table \
  --save_report
```

- 注意：测试模型参数设置应与best模型参数设置相统一

## 4. 输出目录结构

以 `--exp_name exp` 为例：

- `outputs/exp/search_results.csv`：所有 trial 结果表（验证集排序）
- `outputs/exp/best/`：最佳配置权重、曲线、混淆矩阵、摘要
- `outputs/exp/baseline/`：baseline 对照结果
- `outputs/exp/training_comparison.png`：best vs baseline 曲线对比
- `outputs/exp/report_grid.md`：参数搜索相关报告
- `outputs/exp/report_final.md` : 最终报告
- `outputs/exp/test/`：独立测试输出（分组错例、错误表、测试报告）

# 参数搜索实现报告

## 任务与要求对照

- 任务：使用手写反向传播的 NumPy MLP 完成 Fashion-MNIST 10 分类。
- 方法：实现可切换激活函数、SGD/Adam、学习率衰减、L2 正则、验证集选优。
- 评估：测试准确率、混淆矩阵、错例分析、第一层权重可视化。

## 实验设置

- 数据预处理：baseline / flip / mask / flip_mask
- 隐藏层结构：[[128],[256,128],[512,256]]
- 激活函数：ReLU / Tanh / Sigmoid
- Dropout：0.0 / 0.2
- 优化器：SGD / Adam
- 学习率：(0.01,0.001)
- 学习率衰减：(0.95,0.9)
- 权重衰减：(1e-4,5e-4)

## 结果

- 试验总数：20
- 最佳验证准确率：0.8968
- 最佳配置测试准确率：0.8856
- 初始配置测试准确率：0.8879

### Top-K 配置

| trial_id | preprocess_mode | hidden_sizes | activation | dropout | optimizer | learning_rate | lr_decay | weight_decay | best_val_acc | best_epoch | param_count |
| -------: | :-------------- | :----------- | :--------- | ------: | :-------- | ------------: | -------: | -----------: | -----------: | ---------: | ----------: |
|        7 | baseline        | [128]        | tanh       |       0 | adam      |         0.001 |     0.95 |       0.0001 |        0.892 |         28 |      101770 |
|       19 | flip            | [256, 128]   | tanh       |       0 | adam      |         0.001 |     0.95 |       0.0005 |     0.888667 |         24 |      235146 |
|        1 | baseline        | [256, 128]   | tanh       |     0.2 | adam      |         0.001 |     0.95 |       0.0005 |        0.885 |         26 |      235146 |
|        6 | mask            | [256, 128]   | tanh       |     0.2 | adam      |         0.001 |     0.95 |       0.0001 |         0.88 |         30 |      235146 |
|       13 | flip_mask       | [256, 128]   | relu       |     0.2 | adam      |         0.001 |     0.95 |       0.0001 |     0.878333 |         25 |      235146 |

### 最优与基线差异

- preprocess_mode: baseline=baseline -> best=baseline
- hidden_sizes: baseline=[256, 128] -> best=[128]
- activation: baseline=relu -> best=tanh
- dropout: baseline=0.0 -> best=0.0
- optimizer: baseline=sgd -> best=adam
- learning_rate: baseline=0.01 -> best=0.001
- lr_decay: baseline=0.95 -> best=0.95
- weight_decay: baseline=0.0001 -> best=0.0001

## 错误分析

- `Shirt` -> `T-shirt/top`: 131 次（占全部样本 1.31%）
- `Coat` -> `Pullover`: 107 次（占全部样本 1.07%）
- `Shirt` -> `Pullover`: 93 次（占全部样本 0.93%）
- `T-shirt/top` -> `Shirt`: 87 次（占全部样本 0.87%）
- `Pullover` -> `Coat`: 77 次（占全部样本 0.77%）

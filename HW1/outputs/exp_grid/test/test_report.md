# Test Report

- Model: `outputs/exp_grid/best/best_model.npz`
- Preprocess mode: `baseline`
- Test loss: `0.3186`
- Test accuracy: `0.8841`

## Top Confusions

- `Shirt` -> `T-shirt/top`: `127` 次，占全体样本 `1.27%`，原因：上衣类轮廓接近，边缘与纹理差异弱
- `Coat` -> `Pullover`: `103` 次，占全体样本 `1.03%`，原因：上衣类轮廓接近，边缘与纹理差异弱
- `Shirt` -> `Pullover`: `101` 次，占全体样本 `1.01%`，原因：上衣类轮廓接近，边缘与纹理差异弱
- `T-shirt/top` -> `Shirt`: `84` 次，占全体样本 `0.84%`，原因：上衣类轮廓接近，边缘与纹理差异弱
- `Shirt` -> `Coat`: `80` 次，占全体样本 `0.80%`，原因：上衣类轮廓接近，边缘与纹理差异弱

## Error Analysis Notes

- 高频错分集中在相似轮廓类别，说明模型更多依赖粗粒度形状。
- 可通过更强数据增强和更丰富特征表达进一步缓解。

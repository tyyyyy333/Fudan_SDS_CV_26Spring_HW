# Task 1 Augmentation and TTA Ablation Results

This ablation uses the same backbone and optimizer setting across runs:

- Model: pretrained ResNet34
- Training: 30 epochs, AdamW, image size 256
- Dataset split: Oxford-IIIT Pet train/validation split from `configs/task1_aug_ablation.yaml`
- Test-time augmentation is disabled during the augmentation ablation, then evaluated separately with horizontal flip TTA.

## Augmentation Ablation

| setting | best epoch | best val acc | test acc | delta vs no aug |
| --- | ---: | ---: | ---: | ---: |
| RandomResizedCrop | 27 | 0.936141 | 0.919324 | +0.005996 |
| RandomResizedCrop + HFlip | 14 | 0.942935 | 0.918234 | +0.004906 |
| RandomResizedCrop + HFlip + ColorJitter + RandomErasing | 24 | 0.942935 | 0.916326 | +0.002998 |
| No augmentation | 10 | 0.933424 | 0.913328 | +0.000000 |
| RandomResizedCrop + HFlip + ColorJitter | 10 | 0.941576 | 0.911420 | -0.001908 |
| Full recipe + RandAugment | 29 | 0.937500 | 0.902971 | -0.010357 |
| RandomResizedCrop + HFlip + ColorJitter + RandomErasing + MixUp/CutMix | 28 | 0.945652 | 0.896975 | -0.016353 |
| RandomResizedCrop + HFlip + ColorJitter + MixUp/CutMix | 23 | 0.947011 | 0.896157 | -0.017171 |

## Horizontal Flip TTA

| setting | no TTA acc | HFlip TTA acc | TTA delta |
| --- | ---: | ---: | ---: |
| RandomResizedCrop | 0.919324 | 0.920142 | +0.000818 |
| RandomResizedCrop + HFlip | 0.918234 | 0.920959 | +0.002726 |
| RandomResizedCrop + HFlip + ColorJitter + RandomErasing | 0.916326 | 0.917144 | +0.000818 |
| No augmentation | 0.913328 | 0.915236 | +0.001908 |
| RandomResizedCrop + HFlip + ColorJitter | 0.911420 | 0.914963 | +0.003543 |
| Full recipe + RandAugment | 0.902971 | 0.902971 | +0.000000 |
| RandomResizedCrop + HFlip + ColorJitter + RandomErasing + MixUp/CutMix | 0.896975 | 0.902153 | +0.005179 |
| RandomResizedCrop + HFlip + ColorJitter + MixUp/CutMix | 0.896157 | 0.898610 | +0.002453 |

## Takeaways

The strongest training-time augmentation in this controlled 30-epoch sweep is RandomResizedCrop, improving test accuracy by about 0.60 percentage points over the no-augmentation baseline. Adding horizontal flip improves validation accuracy and gives the best final result when paired with horizontal-flip TTA.

MixUp/CutMix and RandAugment are too strong under this short training setup. They can increase validation accuracy in some runs, but they reduce test accuracy substantially, suggesting that they need longer training or retuned learning-rate/regularization settings.

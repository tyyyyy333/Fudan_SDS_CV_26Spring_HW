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

## Stronger TTA and TTT

The stronger TTA setting averages six views: scales 0.875, 1.0, and 1.125, each with and without horizontal flip. TTT uses unlabeled entropy minimization at test time and updates only BatchNorm affine parameters.

| setting | test acc | test loss |
| --- | ---: | ---: |
| RandomResizedCrop + HFlip TTA | 0.920142 | 0.669896 |
| RandomResizedCrop + scale/HFlip TTA | 0.917144 | 0.687769 |
| RandomResizedCrop + HFlip TTA + TTT, lr=1e-3 | 0.922050 | 0.634769 |
| RandomResizedCrop + train HFlip + HFlip TTA | 0.920959 | 0.658422 |
| RandomResizedCrop + train HFlip + scale/HFlip TTA | 0.919052 | 0.672995 |
| RandomResizedCrop + train HFlip + HFlip TTA + TTT, lr=1e-5 | 0.916053 | 0.664087 |
| RandomResizedCrop + train HFlip + HFlip TTA + TTT, lr=1e-4 | 0.918779 | 0.651450 |
| RandomResizedCrop + train HFlip + HFlip TTA + TTT, lr=5e-4 | 0.925048 | 0.630322 |
| RandomResizedCrop + train HFlip + HFlip TTA + TTT, lr=1e-3 | 0.925593 | 0.628310 |
| RandomResizedCrop + train HFlip + HFlip TTA + TTT, lr=2e-3 | 0.923685 | 0.644925 |
| RandomResizedCrop + train HFlip + HFlip TTA + TTT, 2 steps, lr=1e-3 | 0.922595 | 0.641236 |
| RandomResizedCrop + train HFlip + scale/HFlip TTA + TTT, lr=5e-4 | 0.923957 | 0.642208 |

## Takeaways

The strongest training-time augmentation in this controlled 30-epoch sweep is RandomResizedCrop, improving test accuracy by about 0.60 percentage points over the no-augmentation baseline. Adding horizontal flip improves validation accuracy and gives the best final result when paired with horizontal-flip TTA.

MixUp/CutMix and RandAugment are too strong under this short training setup. They can increase validation accuracy in some runs, but they reduce test accuracy substantially, suggesting that they need longer training or retuned learning-rate/regularization settings.

Stronger multi-scale TTA does not help on this setup; it underperforms simple horizontal-flip TTA. TTT is more effective: the best configuration is RandomResizedCrop + train-time HFlip + HFlip TTA + one entropy-minimization step with `lr=1e-3`, reaching 0.925593 test accuracy.

## Focal Loss

We also tested focal loss and a CE+focal mixture on top of the Task 1 ResNet34 setup. The implementation supports both normal hard labels and MixUp/CutMix soft labels.

| setting | train recipe | loss | best val acc | test acc |
| --- | --- | --- | ---: | ---: |
| Crop + HFlip CE baseline | RandomResizedCrop + HFlip | CE | 0.942935 | 0.920959 |
| Crop + HFlip focal gamma=1 | RandomResizedCrop + HFlip | focal | 0.947011 | 0.916871 |
| Crop + HFlip focal gamma=2 | RandomResizedCrop + HFlip | focal | 0.945652 | 0.916599 |
| Crop + HFlip CE+focal | RandomResizedCrop + HFlip | 0.7 CE + 0.3 focal, gamma=2 | 0.942935 | 0.919324 |
| Original full recipe baseline | Crop + HFlip + ColorJitter + RandomErasing + MixUp/CutMix | CE | 0.948370 | 0.898065 |
| Full recipe focal gamma=2 | Crop + HFlip + ColorJitter + RandomErasing + MixUp/CutMix | focal | 0.941576 | 0.901336 |
| Full recipe CE+focal | Crop + HFlip + ColorJitter + RandomErasing + MixUp/CutMix | 0.7 CE + 0.3 focal, gamma=2 | 0.940217 | 0.904061 |

Focal loss does not improve the cleaner crop + HFlip setting. It raises validation accuracy for pure focal, but test accuracy drops by about 0.4 percentage points, so the higher validation score does not transfer. The CE+focal mixture is safer, but still slightly below the CE baseline.

On the original full baseline recipe, focal-style losses help relative to the 89.8% CE full-recipe baseline, with CE+focal reaching 90.41%. However, this mainly mitigates the overly strong MixUp/CutMix/RandomErasing recipe; it does not close the gap to the simpler crop + HFlip recipe. The likely reason is that focal loss further emphasizes hard examples, while this dataset already has noisy hard cases and strong regularization, so it can over-focus on samples where background, pose, or annotation ambiguity is not reliably class-discriminative.

## Baseline Clarification

The original `outputs/task1/baseline_pretrained_resnet34` result and the `aug_none` result in this ablation are different baselines:

| baseline | training recipe | TTA | test acc |
| --- | --- | --- | ---: |
| Original baseline | pretrained ResNet34 + RandomResizedCrop + HFlip + ColorJitter + MixUp/CutMix + RandomErasing, 50 epochs | HFlip | 0.898065 |
| Ablation control | pretrained ResNet34 + resize only, 30 epochs | none | 0.913328 |

The 89.8% original baseline is consistent with the new ablation: the closest recipe, `RandomResizedCrop + HFlip + ColorJitter + RandomErasing + MixUp/CutMix`, reaches 0.896975. This suggests the gap is mainly caused by the strong regularization recipe rather than by an evaluation mismatch. Under the current optimizer and epoch budget, MixUp/CutMix and RandAugment appear too strong for this Task 1 setup.

## Task 3 Segmentation-Guided Classification

We also tested whether the Task 3 segmentation model can improve Task 1 classification. The pipeline first uses the Task 3 `CE + Dice` U-Net checkpoint to predict pet/border/background masks for the Task 1 images, caches those masks, then trains the same pretrained ResNet34 classifier on images where the predicted background is dimmed and the predicted pet + border region is preserved.

| setting | segmentation source | train-time aug | test-time aug | best val acc | test acc |
| --- | --- | --- | --- | ---: | ---: |
| ResNet34 + crop + train HFlip | none | RandomResizedCrop + HFlip | HFlip TTA | 0.942935 | 0.920959 |
| ResNet34 + Task3 pet/border guidance | Task3 `CE + Dice` U-Net | RandomResizedCrop + HFlip | HFlip TTA | 0.945652 | 0.915781 |
| Joint Task1+Task3 soft guidance | Task3 `CE + Dice` U-Net, jointly fine-tuned | Resize + HFlip | HFlip TTA | 0.934783 | 0.916599 |

The segmentation-guided input improves validation accuracy slightly but lowers test accuracy by about 0.52 percentage points relative to the crop + HFlip + HFlip TTA baseline. The most likely reasons are:

- Breed classification can rely on context, pose, fur boundary, scale, and background correlations; aggressively dimming the predicted background may remove useful cues.
- The Task 3 mask is optimized for segmentation mIoU, not classification. Small mask errors around ears, tails, and fur texture can remove class-discriminative details.
- The current guidance is a hard post-processing transform. A softer two-branch model, RGB + mask as an additional channel, or late fusion may preserve more information than dimming pixels before classification.
- The segmentation model was trained separately, so any distribution or resize mismatch between Task 3 mask prediction and Task 1 classification augmentation can introduce noise.

We then tested a joint Task1+Task3 variant. The model initializes Task 3 from the `CE + Dice` checkpoint, predicts a soft pet+border probability map, uses that map to dim the background in a differentiable way, and trains the ResNet34 classifier together with the segmentation branch. The total loss is classification CE plus `0.25 * segmentation_loss`, so classification gradients can also affect the segmentation-guided input.

The joint model is slightly better than the offline hard-mask variant (`0.916599` vs. `0.915781` test accuracy), but it still does not beat the plain crop + HFlip baseline (`0.920959`). Its best validation accuracy is also lower (`0.934783`), while training accuracy reaches 1.0 quickly, suggesting overfitting. This supports the same conclusion: segmentation guidance is useful as an inductive bias, but the current pixel-dimming mechanism removes or downweights information that the breed classifier can use. A less destructive design, such as RGB + soft mask channels, two-branch late fusion, or a learned attention gate on intermediate features, is more promising than directly suppressing image pixels.

## Reproduction Commands

The main commands used for the added Task 1 experiments are:

```bash
python scripts/run_task1_sweep.py --config configs/task1_aug_ablation.yaml

python scripts/run_task1_seg_guided_train.py \
  --config configs/task1_seg_guided.yaml

python scripts/run_task1_task3_joint_train.py \
  --config configs/task1_task3_joint.yaml

python scripts/run_task1_sweep.py \
  --config configs/task1_focal_ablation.yaml

python scripts/run_task1_sweep.py \
  --config configs/task1_focal_full_recipe.yaml
```

The best TTT result was obtained with:

```bash
python scripts/run_task1_eval_checkpoint.py \
  --checkpoint outputs/task1_aug_ablation/aug_crop_flip/best.pt \
  --tta-mode hflip \
  --ttt \
  --ttt-steps 1 \
  --ttt-lr 1e-3
```

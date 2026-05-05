from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights
from torchvision.models.resnet import BasicBlock, ResNet


def _zero_module(module):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.zeros_(module.weight)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_dim = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        _zero_module(self.mlp[2])

    def forward(self, x):
        weights = 2.0 * self.mlp(self.pool(x))
        return x * weights


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_dim = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
        )
        self.activation = nn.Sigmoid()
        _zero_module(self.shared[2])

    def forward(self, x):
        avg_weights = self.shared(self.avg_pool(x))
        max_weights = self.shared(self.max_pool(x))
        return x * (2.0 * self.activation(avg_weights + max_weights))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.activation = nn.Sigmoid()
        _zero_module(self.conv)

    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        attention = 2.0 * self.activation(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attention


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        return self.spatial_attention(x)


class SEBasicBlock(BasicBlock):
    def __init__(self, *args, reduction=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = SEBlock(self.bn2.num_features, reduction=reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class CBAMBasicBlock(BasicBlock):
    def __init__(
        self,
        *args,
        reduction=16,
        spatial_kernel_size=7,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.attention = CBAMBlock(
            self.bn2.num_features,
            reduction=reduction,
            spatial_kernel_size=spatial_kernel_size,
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


RESNET_VARIANTS = {
    "resnet18": {"layers": [2, 2, 2, 2], "weights": ResNet18_Weights.IMAGENET1K_V1, "block": BasicBlock},
    "resnet34": {"layers": [3, 4, 6, 3], "weights": ResNet34_Weights.IMAGENET1K_V1, "block": BasicBlock},
    "se_resnet18": {"layers": [2, 2, 2, 2], "weights": ResNet18_Weights.IMAGENET1K_V1, "block": SEBasicBlock},
    "se_resnet34": {"layers": [3, 4, 6, 3], "weights": ResNet34_Weights.IMAGENET1K_V1, "block": SEBasicBlock},
    "cbam_resnet18": {"layers": [2, 2, 2, 2], "weights": ResNet18_Weights.IMAGENET1K_V1, "block": CBAMBasicBlock},
    "cbam_resnet34": {"layers": [3, 4, 6, 3], "weights": ResNet34_Weights.IMAGENET1K_V1, "block": CBAMBasicBlock},
}

TIMM_HEAD_PREFIXES = ("head.", "classifier.", "fc.", "head_dist.")


def _unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _load_partial_state_dict(model, state_dict):
    current_state = model.state_dict()
    filtered_state = {
        name: tensor
        for name, tensor in state_dict.items()
        if name in current_state and current_state[name].shape == tensor.shape
    }
    model.load_state_dict(filtered_state, strict=False)


def _load_local_weights(model, weights_path):
    checkpoint = torch.load(Path(weights_path).expanduser(), map_location="cpu")
    state_dict = _unwrap_state_dict(checkpoint)
    _load_partial_state_dict(model, state_dict)


def _build_block_factory(block_cls, **block_kwargs):
    if block_cls is BasicBlock:
        return block_cls

    class BlockFactory(block_cls):
        expansion = block_cls.expansion

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, **block_kwargs)

    return BlockFactory


def _resnet_block_callable(variant, model_cfg):
    block = RESNET_VARIANTS[variant]["block"]
    if block is SEBasicBlock:
        return _build_block_factory(
            block,
            reduction=int(model_cfg.get("attention_reduction", 16)),
        )
    if block is CBAMBasicBlock:
        return _build_block_factory(
            block,
            reduction=int(model_cfg.get("attention_reduction", 16)),
            spatial_kernel_size=int(model_cfg.get("attention_spatial_kernel_size", 7)),
        )
    return _build_block_factory(
        block,
    )


def _build_resnet_model(variant, pretrained, num_classes, model_cfg):
    spec = RESNET_VARIANTS[variant]
    block = _resnet_block_callable(variant, model_cfg)
    weights_path = model_cfg.get("weights_path")
    model = ResNet(block, spec["layers"])
    dropout = float(model_cfg.get("dropout", 0.0))
    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    if weights_path:
        _load_local_weights(model, weights_path)
    elif pretrained:
        pretrained_state = spec["weights"].get_state_dict(progress=True)
        _load_partial_state_dict(model, pretrained_state)

    head_names = {name for name, _ in model.named_parameters() if name.startswith("fc.") or ".attention." in name}
    model._head_parameter_names = head_names
    return model


def _timm_head_names(model):
    return {
        name
        for name, _ in model.named_parameters()
        if name.startswith(TIMM_HEAD_PREFIXES)
    }


def _build_timm_model(variant, pretrained, num_classes, model_cfg, image_size=None):
    import timm

    weights_path = model_cfg.get("weights_path")
    kwargs = {
        "pretrained": bool(pretrained and not weights_path),
        "num_classes": num_classes,
        "drop_rate": float(model_cfg.get("dropout", 0.0)),
        "drop_path_rate": float(model_cfg.get("drop_path_rate", 0.0)),
    }
    if image_size is not None:
        kwargs["img_size"] = int(image_size)
    try:
        model = timm.create_model(variant, **kwargs)
    except TypeError:
        kwargs.pop("img_size", None)
        model = timm.create_model(variant, **kwargs)
    if weights_path:
        _load_local_weights(model, weights_path)
    model._head_parameter_names = _timm_head_names(model)
    return model


def build_model(config, num_classes):
    model_cfg = config["model"]
    family = model_cfg.get("family", "resnet")
    variant = model_cfg["variant"]
    pretrained = bool(model_cfg.get("pretrained", True))

    if family == "resnet":
        if variant not in RESNET_VARIANTS:
            raise ValueError(f"Unsupported ResNet variant: {variant}")
        return _build_resnet_model(variant, pretrained, num_classes, model_cfg)

    if family == "timm":
        return _build_timm_model(
            variant,
            pretrained,
            num_classes,
            model_cfg,
            image_size=config.get("data", {}).get("image_size"),
        )

    raise ValueError(f"Unsupported model family: {family}")


def set_backbone_trainable(model, trainable):
    head_names = getattr(model, "_head_parameter_names", set())
    for name, parameter in model.named_parameters():
        if name in head_names:
            continue
        parameter.requires_grad = trainable


def build_optimizer(model, config):
    optimizer_cfg = config["optimizer"]
    head_parameter_names = getattr(model, "_head_parameter_names", set())
    head_params = []
    backbone_params = []
    for name, parameter in model.named_parameters():
        if name in head_parameter_names:
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    parameter_groups = []
    if head_params:
        parameter_groups.append(
            {
                "params": head_params,
                "lr": float(optimizer_cfg["head_lr"]),
                "weight_decay": float(optimizer_cfg.get("weight_decay", 0.0)),
            }
        )
    if backbone_params:
        parameter_groups.append(
            {
                "params": backbone_params,
                "lr": float(optimizer_cfg["backbone_lr"]),
                "weight_decay": float(optimizer_cfg.get("weight_decay", 0.0)),
            }
        )

    optimizer_name = optimizer_cfg.get("name", "adamw").lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameter_groups,
            betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))),
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameter_groups,
            momentum=float(optimizer_cfg.get("momentum", 0.9)),
            nesterov=bool(optimizer_cfg.get("nesterov", True)),
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

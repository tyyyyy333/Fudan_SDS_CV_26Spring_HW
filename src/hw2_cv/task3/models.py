import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(kernel_size=2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_bilinear,
    ):
        super().__init__()
        self.use_bilinear = use_bilinear
        if use_bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=3,
        base_channels=32,
        bottleneck_dropout=0.0,
        use_bilinear=False,
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.inc = DoubleConv(in_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.down4 = DownBlock(c4, c5)
        self.bottleneck_dropout = nn.Dropout2d(p=bottleneck_dropout) if bottleneck_dropout > 0 else nn.Identity()

        self.up1 = UpBlock(c5, c4, c4, use_bilinear=use_bilinear)
        self.up2 = UpBlock(c4, c3, c3, use_bilinear=use_bilinear)
        self.up3 = UpBlock(c3, c2, c2, use_bilinear=use_bilinear)
        self.up4 = UpBlock(c2, c1, c1, use_bilinear=use_bilinear)
        self.outc = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.bottleneck_dropout(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def initialize_model(model, init_cfg):
    if isinstance(init_cfg, str):
        init_name = init_cfg
    else:
        init_name = init_cfg.get("name", "kaiming")

    if init_name != "kaiming":
        raise ValueError(f"Unsupported initialization: {init_name}")

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def build_model(config):
    model_cfg = config["model"]
    model = UNet(
        in_channels=model_cfg.get("in_channels", 3),
        num_classes=model_cfg.get("num_classes", 3),
        base_channels=model_cfg.get("base_channels", 32),
        bottleneck_dropout=float(model_cfg.get("bottleneck_dropout", 0.0)),
        use_bilinear=bool(model_cfg.get("use_bilinear", False)),
    )
    initialize_model(model, model_cfg.get("init", "kaiming"))
    return model

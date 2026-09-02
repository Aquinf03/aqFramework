"""Classic and modern CNN architectures (aq-owned, not torchvision imports)."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import SE, conv1x1, conv3x3


# ── LeNet-5 (LeCun et al.) — works on small or large via adaptive pool ─────────


class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10, in_ch: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 6, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 120, 5),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ── AlexNet (Krizhevsky et al.) ────────────────────────────────────────────────


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, in_ch: int = 3, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, 11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 192, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ── VGG ────────────────────────────────────────────────────────────────────────


_VGG_CFGS: dict[str, list] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [
        64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
        512, 512, 512, 512, "M", 512, 512, 512, 512, "M",
    ],
}


def _make_vgg_layers(cfg: list, in_ch: int, batch_norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    ch = in_ch
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(2, 2))
            continue
        conv = nn.Conv2d(ch, int(v), 3, padding=1)
        if batch_norm:
            layers += [conv, nn.BatchNorm2d(int(v)), nn.ReLU(inplace=True)]
        else:
            layers += [conv, nn.ReLU(inplace=True)]
        ch = int(v)
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(
        self,
        cfg_name: str = "vgg16",
        num_classes: int = 1000,
        in_ch: int = 3,
        batch_norm: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        cfg = _VGG_CFGS[cfg_name]
        self.features = _make_vgg_layers(cfg, in_ch, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ── ResNet ─────────────────────────────────────────────────────────────────────


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = conv1x1(in_ch, out_ch)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch, stride)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = conv1x1(out_ch, out_ch * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: type[nn.Module],
        layers: list[int],
        num_classes: int = 1000,
        in_ch: int = 3,
    ):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._init()

    def _make_layer(self, block: type[nn.Module], planes: int, blocks: int, stride: int = 1):
        down = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            down = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, down)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


# ── Inception-v1 (GoogLeNet) ───────────────────────────────────────────────────


class InceptionBlock(nn.Module):
    def __init__(self, in_ch: int, ch1: int, ch3r: int, ch3: int, ch5r: int, ch5: int, pool_proj: int):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, ch1, 1), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, ch3r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3r, ch3, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.b5 = nn.Sequential(
            nn.Conv2d(in_ch, ch5r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5r, ch5, 5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool_proj, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.pool(x)], 1)


class Inception(nn.Module):
    """GoogLeNet / Inception-v1 style classifier."""

    def __init__(self, num_classes: int = 1000, in_ch: int = 3, dropout: float = 0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.inception3b(self.inception3a(x))
        x = self.maxpool(x)
        x = self.inception4e(
            self.inception4d(self.inception4c(self.inception4b(self.inception4a(x))))
        )
        x = self.maxpool(x)
        x = self.inception5b(self.inception5a(x))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(self.dropout(x))


# ── EfficientNet-B0 (Tan & Le) — MBConv + SE + compound scale entry point ──────


class MBConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        expand: int,
        kernel: int = 3,
        se_ratio: float = 0.25,
    ):
        super().__init__()
        hidden = in_ch * expand
        self.use_skip = stride == 1 and in_ch == out_ch
        layers: list[nn.Module] = []
        if expand != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, kernel, stride=stride, padding=kernel // 2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            SE(hidden, max(1, int(1 / se_ratio)) if se_ratio else 4),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return x + out if self.use_skip else out


def _eff_stage(in_ch: int, out_ch: int, n: int, stride: int, expand: int, kernel: int = 3) -> nn.Sequential:
    blocks = [MBConv(in_ch, out_ch, stride, expand, kernel)]
    for _ in range(1, n):
        blocks.append(MBConv(out_ch, out_ch, 1, expand, kernel))
    return nn.Sequential(*blocks)


class EfficientNet(nn.Module):
    """EfficientNet-B0 layout (scalable via width/depth multipliers)."""

    def __init__(
        self,
        num_classes: int = 1000,
        in_ch: int = 3,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        dropout: float = 0.2,
    ):
        super().__init__()

        def c(ch: int) -> int:
            return max(8, int(ch * width_mult) // 8 * 8)

        def d(n: int) -> int:
            return max(1, int(round(n * depth_mult)))

        stem_ch = c(32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stem_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.SiLU(inplace=True),
        )
        # (out, num, stride, expand, kernel) — B0
        cfg = [
            (16, 1, 1, 1, 3),
            (24, 2, 2, 6, 3),
            (40, 2, 2, 6, 5),
            (80, 3, 2, 6, 3),
            (112, 3, 1, 6, 5),
            (192, 4, 2, 6, 5),
            (320, 1, 1, 6, 3),
        ]
        stages = []
        ch = stem_ch
        for out, n, stride, expand, k in cfg:
            out_c = c(out)
            stages.append(_eff_stage(ch, out_c, d(n), stride, expand, k))
            ch = out_c
        self.blocks = nn.Sequential(*stages)
        head_ch = c(1280)
        self.head = nn.Sequential(
            nn.Conv2d(ch, head_ch, 1, bias=False),
            nn.BatchNorm2d(head_ch),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(head_ch, num_classes),
        )
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


# ── ConvNeXt (Liu et al.) ──────────────────────────────────────────────────────


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)
        self.drop = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dw(x)
        x = x.permute(0, 2, 3, 1)  # NHWC for LayerNorm
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        if self.training and self.drop > 0:
            keep = 1.0 - self.drop
            mask = torch.empty(x.shape[0], 1, 1, 1, device=x.device).bernoulli_(keep) / keep
            x = x * mask
        return shortcut + x


class ConvNeXt(nn.Module):
    def __init__(
        self,
        depths: list[int] | None = None,
        dims: list[int] | None = None,
        num_classes: int = 1000,
        in_ch: int = 3,
        drop_path: float = 0.0,
    ):
        super().__init__()
        depths = depths or [3, 3, 9, 3]
        dims = dims or [96, 192, 384, 768]
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dims[0], 4, stride=4),
            _ChannelsFirstLayerNorm(dims[0]),
        )
        dp_rates = torch.linspace(0, drop_path, sum(depths)).tolist()
        stages = nn.ModuleList()
        downs = nn.ModuleList()
        cur = 0
        for i in range(4):
            blocks = [ConvNeXtBlock(dims[i], float(dp_rates[cur + j])) for j in range(depths[i])]
            stages.append(nn.Sequential(*blocks))
            cur += depths[i]
            if i < 3:
                downs.append(
                    nn.Sequential(
                        _ChannelsFirstLayerNorm(dims[i]),
                        nn.Conv2d(dims[i], dims[i + 1], 2, stride=2),
                    )
                )
        self.stages = stages
        self.downs = downs
        self.norm = _ChannelsFirstLayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        x = self.norm(x)
        x = x.mean([-2, -1])
        return self.head(x)


class _ChannelsFirstLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

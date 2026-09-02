"""Build aq-owned CNN by recipe arch name."""

from __future__ import annotations

from torch import nn

from . import models as M

# public name → (ctor kwargs builder)
ARCH_ALIASES: dict[str, str] = {
    "lenet": "lenet",
    "lenet5": "lenet",
    "alexnet": "alexnet",
    "vgg": "vgg16",
    "vgg11": "vgg11",
    "vgg13": "vgg13",
    "vgg16": "vgg16",
    "vgg19": "vgg19",
    "resnet": "resnet18",
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "resnet152": "resnet152",
    "inception": "inception",
    "googlenet": "inception",
    "inception-v1": "inception",
    "efficientnet": "efficientnet_b0",
    "efficientnet-b0": "efficientnet_b0",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet-b1": "efficientnet_b1",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet-b2": "efficientnet_b2",
    "efficientnet_b2": "efficientnet_b2",
    "convnext": "convnext_tiny",
    "convnext-tiny": "convnext_tiny",
    "convnext_tiny": "convnext_tiny",
    "convnext-small": "convnext_small",
    "convnext_small": "convnext_small",
    "convnext-base": "convnext_base",
    "convnext_base": "convnext_base",
}


def list_arches() -> list[str]:
    return sorted(set(ARCH_ALIASES.values()))


def build_cnn(arch: str, num_classes: int, in_ch: int = 3) -> nn.Module:
    key = ARCH_ALIASES.get(str(arch).lower().replace(" ", ""))
    if not key:
        raise SystemExit(
            f"unknown cnn arch {arch!r}. Supported: {', '.join(list_arches())}"
        )
    if key == "lenet":
        return M.LeNet(num_classes=num_classes, in_ch=in_ch)
    if key == "alexnet":
        return M.AlexNet(num_classes=num_classes, in_ch=in_ch)
    if key.startswith("vgg"):
        return M.VGG(cfg_name=key, num_classes=num_classes, in_ch=in_ch)
    if key == "resnet18":
        return M.ResNet(M.BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_ch=in_ch)
    if key == "resnet34":
        return M.ResNet(M.BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_ch=in_ch)
    if key == "resnet50":
        return M.ResNet(M.Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_ch=in_ch)
    if key == "resnet101":
        return M.ResNet(M.Bottleneck, [3, 4, 23, 3], num_classes=num_classes, in_ch=in_ch)
    if key == "resnet152":
        return M.ResNet(M.Bottleneck, [3, 8, 36, 3], num_classes=num_classes, in_ch=in_ch)
    if key == "inception":
        return M.Inception(num_classes=num_classes, in_ch=in_ch)
    if key == "efficientnet_b0":
        return M.EfficientNet(num_classes=num_classes, in_ch=in_ch, width_mult=1.0, depth_mult=1.0)
    if key == "efficientnet_b1":
        return M.EfficientNet(num_classes=num_classes, in_ch=in_ch, width_mult=1.0, depth_mult=1.1, dropout=0.2)
    if key == "efficientnet_b2":
        return M.EfficientNet(num_classes=num_classes, in_ch=in_ch, width_mult=1.1, depth_mult=1.2, dropout=0.3)
    if key == "convnext_tiny":
        return M.ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=num_classes, in_ch=in_ch)
    if key == "convnext_small":
        return M.ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], num_classes=num_classes, in_ch=in_ch)
    if key == "convnext_base":
        return M.ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], num_classes=num_classes, in_ch=in_ch)
    raise SystemExit(f"cnn arch {key!r} not wired")


def default_image_size(arch: str) -> int:
    key = ARCH_ALIASES.get(str(arch).lower().replace(" ", ""), str(arch).lower())
    if key == "lenet":
        return 32
    if key == "inception":
        return 224
    if key.startswith("efficientnet_b1"):
        return 240
    if key.startswith("efficientnet_b2"):
        return 260
    return 224

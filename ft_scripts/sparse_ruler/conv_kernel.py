import os
import torch
import torch.nn as nn


DEFAULT_OUT_PATH = (
    "/mnt/oos/yichengfu/Innovator-lm-evaluation-hardness/"
    "x-attention-main/xattn/conv_weights/conv_kernel_7x7.pt"
)


def make_vertical_plus_diag_kernel(size: int = 7) -> torch.Tensor:
    """
    初始化单个 7x7 kernel：
    1. 中间竖线为 1
    2. 左上到右下主对角线为 1
    3. 其余为 0

    返回 shape: [7, 7]
    """
    weight = torch.zeros(size, size, dtype=torch.float32)
    center = size // 2
    weight[:, center] = 1.0
    for i in range(size):
        weight[i, i] = 1.0
    return weight


def make_layer_head_kernel(
    num_layers: int = 32,
    num_heads: int = 32,
    kernel_size: int = 7,
) -> torch.Tensor:
    """
    返回 shape:
        [num_layers, num_heads, 7, 7]
    """
    base = make_vertical_plus_diag_kernel(kernel_size)
    weight = base[None, None, :, :].repeat(num_layers, num_heads, 1, 1)
    return weight.contiguous()


class TrainableLayerHeadConvKernel7x7(nn.Module):
    """
    只训练这个参数：

        self.weight: [32, 32, 7, 7]

    其中：
        weight[layer_idx, head_idx] 是对应 layer/head 的 7x7 kernel。
    """

    def __init__(
        self,
        init_path: str | None = None,
        num_layers: int = 32,
        num_heads: int = 32,
        kernel_size: int = 7,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.kernel_size = kernel_size

        if init_path is not None and os.path.exists(init_path):
            weight = torch.load(init_path, map_location="cpu", weights_only=True).float()

            # 兼容旧格式
            if tuple(weight.shape) == (1, 1, kernel_size, kernel_size):
                weight = weight[0, 0][None, None, :, :].repeat(
                    num_layers, num_heads, 1, 1
                )

            # 兼容 [L,H,1,7,7]
            elif weight.dim() == 5 and tuple(weight.shape[-3:]) == (1, kernel_size, kernel_size):
                weight = weight[:, :, 0, :, :]

            elif tuple(weight.shape) != (num_layers, num_heads, kernel_size, kernel_size):
                raise ValueError(
                    f"Expected {(num_layers, num_heads, kernel_size, kernel_size)}, "
                    f"or old [1,1,7,7] / [L,H,1,7,7], got {tuple(weight.shape)}"
                )
        else:
            weight = make_layer_head_kernel(
                num_layers=num_layers,
                num_heads=num_heads,
                kernel_size=kernel_size,
            )

        self.weight = nn.Parameter(weight.contiguous())

    def forward(self):
        return self.weight

    def get_layer_weight(self, layer_idx: int) -> torch.Tensor:
        """
        返回某一层所有 head 的 kernel：

            [32, 7, 7]
        """
        return self.weight[layer_idx]

    @torch.no_grad()
    def save(self, path: str = DEFAULT_OUT_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.weight.detach().float().cpu().contiguous(), path)

    @torch.no_grad()
    def print_kernel(self, layer_idx: int = 0, head_idx: int = 0):
        print(f"kernel[layer={layer_idx}, head={head_idx}]:")
        print(self.weight.detach().float().cpu()[layer_idx, head_idx])
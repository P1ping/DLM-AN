import torch
from torch import nn, Tensor
from torch.nn import Conv2d

LRELU_SLOPE = 0.1


class ResBlock1d(torch.nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=0, groups=8, kernel_size=3):
        super().__init__()
        # If time_emb_dim > 0, this module is time-dependent
        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out)) if time_emb_dim > 0 else None

        self.block1 = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, kernel_size, padding=kernel_size // 2),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv1d(dim_out, dim_out, kernel_size, padding=kernel_size // 2),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb=None):
        """
        Args:
            x (Tensor): [N, T, D].
            mask (Tensor): [N, T].
            time_emb (Tensor): [N, D].
        Returns:
            x (Tensor): [N, T, D].
        """
        x = x.transpose(1, 2)  # [N, D, T]
        mask = mask.unsqueeze(1)  # [N, 1, T]
        h = self.block1(x * mask)  # [N, D, T]
        if time_emb is not None or self.mlp is not None:
            assert time_emb is not None, "time_emb must be provided if mlp is defined (time_emb_dim > 0)"
            assert self.mlp is not None, "mlp must be defined (time_embed_dim > 0) if time_emb is provided"
            h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h * mask)
        output = h + self.res_conv(x * mask)
        output = output * mask
        return output.transpose(1, 2)  # [N, T, D]


class ResBlock2d(nn.Module):
    def __init__(self, channels: int, kernel: int = 3):
        """
        Args:
            channels (int): The number of input channels.
            out_channels (int, optional): The number of output channels. Defaults to `channels`.
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.LeakyReLU(LRELU_SLOPE),
            Conv2d(channels, channels, kernel, padding=kernel // 2),
            nn.LeakyReLU(LRELU_SLOPE),
            Conv2d(channels, channels, kernel, padding=kernel // 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """[BCHW] -> [BCHW]"""
        h = self.layers(x)
        return x + h


class PostNet(nn.Module):
    def __init__(self, D_bridge: int, D_conv2d: int, D_out: int):
        super().__init__()
        self.bridge_linear = nn.Linear(D_out, D_bridge * D_out)
        self.bridge_conv2d = Conv2d(D_bridge + 1, D_conv2d, 3, 1, 1)
        self.resblock_2d = nn.Sequential(ResBlock2d(D_conv2d, kernel=3), ResBlock2d(D_conv2d, kernel=3))
        self.last_conv2d = nn.Sequential(nn.LeakyReLU(LRELU_SLOPE), Conv2d(D_conv2d, 1, 3, 1, 1))

        self.D_bridge = D_bridge
        self.D_out = D_out

    def forward(self, x: Tensor, x_res: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): [N, T, D_out].
            x_res (Tensor): [N, T, D_out].
            mask (Tensor): [N, T], attention mask, True for valid positions.
        Returns:
            x (Tensor): [N, T, D_out].
        """
        x = self.bridge_linear.forward(x)  # [N, T, D_bridge * D_mel]
        x = torch.unflatten(x, dim=-1, sizes=(self.D_bridge, self.D_out))  # [N, T, D_bridge, D_mel]
        x = x.transpose(-2, -3).contiguous()  # [N, D_bridge, T, D_mel]
        x = torch.cat([x_res.unsqueeze(-3), x], dim=-3)  # [N, D_bridge + 2, T, D_mel]

        x = self.bridge_conv2d.forward(x)
        x = self.resblock_2d.forward(x)
        x = self.last_conv2d.forward(x)  # [N, 1, T, D_mel]

        return x.squeeze(-3) * mask.unsqueeze(-1)  # [N, T, D_mel]

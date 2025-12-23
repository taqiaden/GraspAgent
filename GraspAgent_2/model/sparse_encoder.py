import spconv.pytorch as spconv
import torch
from torch import nn


class SparseEncoderIN(nn.Module):
    def __init__(self, in_ch=3, out_ch=512):
        super().__init__()

        self.net = spconv.SparseSequential(
            # 100³ → 100³
            spconv.SubMConv3d(in_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),

            # 100³ → 50³
            spconv.SparseConv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),

            # 50³ → 25³
            spconv.SparseConv3d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),

            # 25³ → 13³
            spconv.SparseConv3d(256, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):

        x=self.net(x)
        x=x.dense()
        # avg = x.mean(dim=[2, 3, 4])
        x = x.amax(dim=[2, 3, 4])
        # x = torch.cat([avg, mx], dim=1)  # (B, 256)
        return x


class SparseResidualBlock(nn.Module):
    """Residual block for sparse tensor: SubMConv3d + InstanceNorm + LeakyReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv1 = spconv.SubMConv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        # self.norm1 = nn.InstanceNorm1d(out_ch, affine=True)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = spconv.SubMConv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        # self.norm2 = nn.InstanceNorm1d(out_ch, affine=True)
        self.act2 = nn.LeakyReLU(0.2)

        if in_ch != out_ch:
            # 1x1 conv to match channels for residual
            self.downsample = spconv.SubMConv3d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.downsample = None

    def forward(self, x: spconv.SparseConvTensor):
        identity = x
        out = self.conv1(x)
        out.features = (out.features.unsqueeze(2)).squeeze(2)
        out = self.act1(out)
        out = self.conv2(out)
        out.features = (out.features.unsqueeze(2)).squeeze(2)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out.features = out.features + identity.features  # residual connection
        out = self.act2(out)
        return out


class SparseEncoderResUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, out_ch=128):
        super().__init__()

        self.stage1 = SparseResidualBlock(in_ch, base_ch)  # 100³
        self.down1 = spconv.SparseSequential(
            spconv.SparseConv3d(base_ch, base_ch*2, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.InstanceNorm1d(base_ch*2, affine=True),
            nn.LeakyReLU(0.2)
        )  # 50³

        self.stage2 = SparseResidualBlock(base_ch*2, base_ch*2)
        self.down2 = spconv.SparseSequential(
            spconv.SparseConv3d(base_ch*2, base_ch*4, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.InstanceNorm1d(base_ch*4, affine=True),
            nn.LeakyReLU(0.2)
        )  # 25³

        self.stage3 = SparseResidualBlock(base_ch*4, base_ch*4)
        self.down3 = spconv.SparseSequential(
            spconv.SparseConv3d(base_ch*4, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.InstanceNorm1d(out_ch, affine=True),
            nn.LeakyReLU(0.2)
        )  # 13³

    def forward(self, x: spconv.SparseConvTensor):
        x1 = self.stage1(x)
        x2 = self.stage2(self.down1(x1))
        x3 = self.stage3(self.down2(x2))
        x4 = self.down3(x3)

        # Convert sparse to dense for global pooling
        dense = x4.dense()
        avg = dense.mean(dim=[2,3,4])
        mx = dense.amax(dim=[2,3,4])
        out = torch.cat([avg, mx], dim=1)  # (B, 2*out_ch)
        return out
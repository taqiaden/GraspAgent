import numpy as np
import spconv.pytorch as spconv
import torch
from torch import nn
import torch.nn.functional  as F
class Encoder2D_IN(nn.Module):
    def __init__(self, in_ch=3, out_ch=512):
        super().__init__()

        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=True),
                # nn.InstanceNorm2d(cout),
                nn.SiLU(),
            )

        self.net = nn.Sequential(
            block(100, 128, 1),
            block(128, 256, 2),
            block(256, 512, 2),
            block(512, out_ch, 2),
        )

        self.head = nn.Sequential(
            # nn.LayerNorm(out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        x = x.dense()
        x = (x != 0).any(dim=1, keepdim=True).float()
        assert x.shape[1]==1, f'{x.shape[1]}'
        x=x[:,0]
        # x: [B, C, H, W]
        x = self.net(x)                  # -> [B, C, H', W']
        x = torch.amax(x, dim=[2, 3])    # global spatial pooling -> [B, C]
        return self.head(x)


class Encoder3D_IN(nn.Module):
    def __init__(self, in_ch=1, out_ch=512):
        super().__init__()

        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.InstanceNorm3d(cout, affine=True),
                nn.SiLU(inplace=True),
            )

        self.net = nn.Sequential(
            block(in_ch, 64, 1),
            block(64, 128, 2),
            block(128, 256, 2),
            block(256, out_ch, 2),
        )

        self.head = nn.Sequential(
            # nn.LayerNorm(out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        x = x.dense()
        x = (x != 0).any(dim=1, keepdim=True).float()

        x = self.net(x)                      # [B, C, D, H, W]

        x = torch.amax(x, dim=[2, 3, 4])     # global spatial pooling
        return self.head(x)

class SparseEncoderIN(nn.Module):
    def __init__(self, in_ch=3, out_ch=512):
        super().__init__()

        def block(cin, cout, stride):
            return spconv.SparseSequential(
                spconv.SparseConv3d(cin, cout, 3, stride=stride, padding=1, bias=True),
                # spconv.SparseBatchNorm(cout),
                # nn.LayerNorm(cout),
                spconv.SparseReLU(),
                # nn.SiLU()
            )

        self.net = spconv.SparseSequential(
            block(in_ch, 64, 1),
            block(64, 128, 2),
            block(128, 256, 2),
            block(256, out_ch, 2),
        )

        self.head = nn.Sequential(
            # nn.LayerNorm(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.dense()
        x = torch.amax(x, dim=[2, 3, 4])  # smoother than amax
        # x=F.normalize(x,p=2,dim=-1,eps=1e-7)
        return self.head(x)

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
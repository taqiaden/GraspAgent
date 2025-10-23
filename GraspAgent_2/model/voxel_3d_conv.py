import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelBackbone(nn.Module):
    def __init__(self, activation=nn.SiLU()):
        super(VoxelBackbone, self).__init__()
        """
        3D CNN for voxel classification.

        Args:
            activation: activation function (default: SiLU).
                        Example: nn.LeakyReLU(0.2), nn.GELU(), etc.
        """

        self.activation = activation

        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1)   # [50,50,50]
        self.in1 = nn.InstanceNorm3d(16)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)  # [25,25,25]
        self.in2 = nn.InstanceNorm3d(32)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)  # [13,13,13]
        self.in3 = nn.InstanceNorm3d(64)

        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1) # [7,7,7]
        self.in4 = nn.InstanceNorm3d(128)

        self.conv5 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1) # [4,4,4]
        self.in5 = nn.InstanceNorm3d(256)

        # Replace AdaptiveAvgPool3d with Conv3d
        self.conv_reduce = nn.Conv3d(256, 256, kernel_size=4, stride=1, padding=0)  # → [N,256,1,1,1]


    def forward(self, x):
        act = self.activation

        x = act(self.in1(self.conv1(x)))
        x = act(self.in2(self.conv2(x)))
        x = act(self.in3(self.conv3(x)))
        x = act(self.in4(self.conv4(x)))
        x = act(self.in5(self.conv5(x)))

        x = self.conv_reduce(x)  # [N,256,1,1,1]
        x = x.view(x.size(0), -1)  # [N,256]

        return x


def number_of_parameters(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

if __name__ == "__main__":
    model = VoxelBackbone()
    x = torch.randn(2, 1, 100, 100, 100)  # batch of 2 voxel grids
    y = model(x)
    print(number_of_parameters(model))
    print(y.shape)  # -> torch.Size([2, 1])

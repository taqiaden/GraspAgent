import torch
import torch.nn as nn
import torch.nn.functional as F

from GraspAgent_2.model.voxel_3d_conv import number_of_parameters


class ConvFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, activation=nn.ReLU(), use_norm=True):
        super().__init__()
        self.activation = activation

        def conv_block(in_ch, out_ch, k=3, s=1, p=1):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)]
            if use_norm:
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            layers.append(self.activation)
            return nn.Sequential(*layers)

        # Backbone
        self.layers = nn.Sequential(
            conv_block(in_channels, 32,s=2,k=7),  # [32, 140, 140]
            conv_block(32, 64, s=2),  # [64, 70, 70]
            conv_block(64, 128, s=2),  # [128, 35, 35]
            conv_block(128, 256, s=2),  # [256, 18, 18]
            conv_block(256, 64, s=2),  # [512, 9, 9]
        )

        # Global pooling → flatten
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # [B, 512, 1, 1]
        # self.fc = nn.Linear(512, feature_dim)  # Final embedding

    def forward(self, x):
        x = self.layers(x)
        x = self.global_pool(x)
        # x = torch.flatten(x, 1)  # [B, 512]
        # x = self.fc(x)  # [B, feature_dim]
        return x


# Example usage
if __name__ == "__main__":
    model = ConvFeatureExtractor(
        in_channels=1,
        activation=nn.LeakyReLU(0.1),  # Try different activations
        use_norm=False
    )
    x = torch.randn(2, 1, 140, 140)
    feat = model(x)
    print("Output shape:", feat.shape)  # [4, 256]
    print(number_of_parameters(model))


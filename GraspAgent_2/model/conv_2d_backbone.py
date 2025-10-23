from torch import nn


class conv_140(nn.Module):
    def __init__(self, in_channels=1, use_norm=True, feat_dim=256):
        """
        in_channels: 1=depth, 3=RGB
        use_norm: whether to apply InstanceNorm
        feat_dim: output feature dimension
        """
        super().__init__()
        self.use_norm = use_norm

        def block(in_ch, out_ch):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)]
            if self.use_norm:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # CNN backbone
        self.conv_layers = nn.Sequential(
            block(in_channels, 32),   # [B, 32, 70, 70]
            block(32, 64),            # [B, 64, 35, 35]
            block(64, 128),           # [B, 128, 18, 18]
            block(128, 256),          # [B, 256, 9, 9]
            nn.AdaptiveAvgPool2d((1, 1))  # [B, 256, 1, 1]
        )

        # Final projection to feature vector
        self.proj = nn.Linear(256, feat_dim)

    def forward(self, x):
        # x: [B, C, 140, 140]
        feat = self.conv_layers(x)          # [B, 256, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [B, 256]
        feat = self.proj(feat)              # [B, feat_dim]
        return feat
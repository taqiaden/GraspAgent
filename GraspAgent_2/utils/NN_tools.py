from torch import nn
import torch.nn.functional as F

def get_auto_groupnorm(num_channels, max_groups=8,affine=True):
    # Find the largest number of groups <= max_groups that divides num_channels
    for g in reversed(range(1, max_groups + 1)):
        if num_channels % g == 0:
            return nn.GroupNorm(num_groups=g, num_channels=num_channels, affine=affine).to('cuda')
    # fallback to LayerNorm behavior
    return nn.GroupNorm(num_groups=1, num_channels=num_channels, affine=affine).to('cuda')

def replace_instance_with_groupnorm(module, max_groups=8,affine=True):
    for name, child in module.named_children():
        if isinstance(child, nn.InstanceNorm2d):
            gn = get_auto_groupnorm(child.num_features, max_groups=max_groups,affine=affine)
            setattr(module, name, gn)
        else:
            replace_instance_with_groupnorm(child, max_groups=max_groups)


class WSConv2d(nn.Conv2d):
    """
    Weight-Standardized Conv2d
    """
    def forward(self, x):
        # Get weight
        w = self.weight
        # Compute per-output-channel mean and std
        mean = w.mean(dim=(1,2,3), keepdim=True)
        std = w.std(dim=(1,2,3), keepdim=True) + 1e-5
        # Standardize
        w_hat = (w - mean) / std
        # Perform convolution with standardized weight
        return F.conv2d(x, w_hat, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

def replace_conv_with_wsconv(module):
    """
    Recursively replace all nn.Conv2d layers with WSConv2d
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            # Keep the same initialization parameters
            ws_conv = WSConv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(child.bias is not None)
            ).to('cuda')
            # Copy the original weights and bias
            ws_conv.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                ws_conv.bias.data.copy_(child.bias.data)
            setattr(module, name, ws_conv)
        else:
            # Recursively replace in child modules
            replace_conv_with_wsconv(child)




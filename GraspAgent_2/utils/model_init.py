import math

from torch import nn

def kaiming_init_all(model, negative_slope=0.2):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=negative_slope, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def init_norm_free_resunet(model):
    """
    Initialize ResUNet:
    - All conv layers: Kaiming Uniform, bias=0
    - Exception: 'c2' conv under 'residual_block' modules: weight=0, bias=0
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Check if parent module is a residual_block and this conv is 'c2'
            # Walk through the hierarchy to see if the parent is residual_block
            parent_names = name.split('.')[:-1]  # remove last part (conv itself)
            is_c2_in_resblock = False
            if parent_names:
                parent = model
                for pn in parent_names:
                    parent = getattr(parent, pn)
                if parent.__class__.__name__.lower() == "residual_block" and name.endswith("c2"):
                    is_c2_in_resblock = True

            if is_c2_in_resblock:
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # print(f"✅ Zero-initialized {name}")
            else:
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='linear')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # print(f"✅ Kaiming-initialized {name}")

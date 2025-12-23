import math

from torch import nn

def gan_init_with_norms(m):
    # conv layers
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if m.weight is not None:
            m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()

    # instance norm
    elif isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
        if m.weight is not None:
            m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.zero_()

    # layer norm
    elif isinstance(m, nn.LayerNorm):
        if m.weight is not None:
            m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.zero_()
def init_resunet(model, init_type='kaiming', a=0.01):
    """
    Initialize ResUNet with LeakyReLU

    Args:
        model: ResUNet model
        init_type: 'kaiming' or 'xavier'
        a: negative slope for LeakyReLU
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=a, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.ConvTranspose2d):
            # Initialize transposed conv with slightly smaller weights
            nn.init.kaiming_normal_(m.weight, a=a, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=a, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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



def orthogonal_init_all(model, gain=1.0):
    """
    Apply orthogonal initialization to all Conv and Linear layers in a model.

    Args:
        model: torch.nn.Module — your model.
        gain: float — scaling factor (e.g., use 0.7–1.0 for SpectralNorm models,
                      or torch.nn.init.calculate_gain('leaky_relu', 0.2) if LeakyReLU is used).
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            # If you have these (even if unused), set neutral initialization
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def init_orthogonal(m,scale=None,gain=1.0):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight,gain=gain)
        if scale is not None:m.weight.data*=scale
        if m.bias is not None:
            nn.init.zeros_(m.bias)
def init_simplex_diversity(m):
    if isinstance(m, nn.Linear) and m.out_features == 128:   # the layer *before* Softmax
        nn.init.normal_(m.weight, mean=0.0, std=0.01)      # small variance
        nn.init.zeros_(m.bias)
def scaled_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        fan_in = m.weight.data.size(1)
        nn.init.orthogonal_(m.weight)
        m.weight.data *= 1.0 / fan_in**0.5
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_weights_xavier(m,gain=1.0):
    """
    Apply Xavier initialization to all linear and convolutional layers
    """
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_uniform_(m.weight,gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_weights_he(m):
    """
    Apply He (Kaiming) initialization to all linear and convolutional layers
    """
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_weights_xavier_normal(m,gain=1.0,scale=1.0):
    """
    Apply Xavier normal initialization
    """
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_normal_(m.weight, gain=gain)
        if scale is not None: m.weight.data *= scale
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_weights_he_normal(m,scale=None):
    """
    Apply He normal initialization
    """
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in',nonlinearity='leaky_relu')
        if scale is not None: m.weight.data *= scale
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def scaled_he_init_all(module, scale=0.1):
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            m.weight.data.mul_(scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def init_weights_normal(m):
    """
    Initialize all weights in the module with a zero-centered Gaussian (mean=0, std=0.02).
    Biases are initialized to zero.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        # if m.bias is not None:
        #     nn.init.constant_(m.bias, 0)
    # elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
    #     nn.init.normal_(m.weight, mean=1.0, std=0.02)  # Norm layers usually start with weight=1
    #     nn.init.constant_(m.bias, 0)


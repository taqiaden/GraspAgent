import torch


def apply_static_spectral_norm(model):
    """
    Manually normalize weights of each Conv2d or Linear layer
    by dividing them by their largest singular value (spectral norm).
    """
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            W = module.weight.data
            # Flatten to 2D matrix [out_features, in_features]
            W_mat = W.view(W.shape[0], -1)

            # Compute spectral norm (largest singular value)
            try:
                sigma = torch.linalg.svdvals(W_mat)[0]  # most accurate
            except RuntimeError:
                # fallback if linalg.svdvals fails (e.g., CUDA)
                u, s, v = torch.svd(W_mat)
                sigma = s[0]

            # Normalize weight
            module.weight.data = W / sigma

            # print(f"Applied static SN to: {name}, σ={sigma.item():.4f}")

def scale_all_weights(model, divisor):
    """
    Divide weights of all layers by a given divisor.
    Works for Conv, Linear, and other modules with a `.weight` tensor.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.div_(divisor)
                print(f"Scaled weights in {name} by 1/{divisor}")


def clip_weight_scales(model, min_std=0.01, max_std=0.1):
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight.data
                std = w.std().item()
                if std < min_std:
                    w.mul_(min_std / (std + 1e-8))
                elif std > max_std:
                    w.mul_(max_std / std)


def fix_weight_scales(model, target_std=1.0, eps=1e-8):
    """
    Normalize each layer's weights to have std = target_std.
    Use after pretraining or normalization removal.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight.data
                current_std = w.std().item()
                if current_std > eps:
                    scale = target_std / current_std
                    w.mul_(scale)
                    print(f"{name}: scaled weights by {scale:.3f}")
                    print('max value', w.abs().max())


def scale_module_weights(module, scale, scale_bias=False):
    """
    Scales all weights in a module (and optionally biases) by a given factor.
    Works recursively for all submodules.
    """
    with torch.no_grad():
        for name, m in module.named_modules():
            if hasattr(m, "weight") and m.weight is not None:
                m.weight.mul_(scale)
                # print(f"Scaled {name}.weight by {scale}")
            if scale_bias and hasattr(m, "bias") and m.bias is not None:
                m.bias.mul_(scale)
                # print(f"Scaled {name}.bias by {scale}")

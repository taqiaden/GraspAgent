import torch
from torch import nn
import torch.nn.functional as F

class LearnableRBFEncoding2D(nn.Module):
    def __init__(self, num_centers=16, init_sigma=0.1,with_layer_norm=True):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(0, 1, num_centers)).to('cuda')
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_sigma))).to('cuda')

        self.norm = nn.LayerNorm(num_centers).to('cuda') if with_layer_norm else None

    def forward(self, x):
        B, C, W, H = x.shape

        x_exp = x.unsqueeze(2)
        diff = x_exp - self.centers.view(1, 1, -1, 1, 1)
        sigma = torch.exp(self.log_sigma)
        rbf = torch.exp(-0.5 * (diff / sigma) ** 2)

        rbf = rbf.reshape(B, C * len(self.centers), W, H)
        return rbf


def depth_sin_cos_encoding(depth, n_freqs=5, max_freq=10.0):
    """
    depth: [B, 1, H, W] tensor (recommended normalized to [0, 1])
    n_freqs: number of frequency bands
    max_freq: maximum frequency multiplier (e.g. 10 or 2π)
    returns: [B, 2*n_freqs, H, W]
    """
    # frequencies spaced geometrically between 1 and max_freq
    freqs = torch.linspace(1.0, max_freq, n_freqs, device=depth.device)
    # expand for broadcasting
    depth_expanded = depth.unsqueeze(2)  # [B, 1, 1, H, W]
    freqs = freqs.view(1, 1, n_freqs, 1, 1)

    # compute sin and cos
    enc_sin = torch.sin(depth_expanded * freqs * torch.pi)
    enc_cos = torch.cos(depth_expanded * freqs * torch.pi)

    # concatenate and reshape to [B, 2*n_freqs, H, W]
    encoding = torch.cat([enc_sin, enc_cos], dim=2)
    encoding = encoding.view(depth.shape[0], -1, depth.shape[2], depth.shape[3])

    return encoding

class LearnableRBFEncoding1d(nn.Module):
    def __init__(self, num_centers=16, init_sigma=0.1, device='cuda',use_ln=True):
        super().__init__()
        self.num_centers = num_centers
        self.register_parameter(
            "centers",
            nn.Parameter(torch.linspace(0, 1, num_centers, device=device))
        )
        self.register_parameter(
            "log_sigma",
            nn.Parameter(torch.log(torch.tensor(init_sigma, device=device)))
        )


        self.ln = nn.LayerNorm(num_centers).to('cuda') if use_ln else None

    def forward(self, x):
        # x shape: [*, D] (any number of leading dims)
        *prefix, D = x.shape

        # Expand for broadcasting
        x_exp = x.unsqueeze(-1)  # [*, D, 1]
        centers = self.centers.view(*([1] * (len(prefix) + 1)), -1)  # [1,...,1, num_centers]

        # Compute Gaussian RBF encoding
        sigma = torch.exp(self.log_sigma)

        diff = x_exp - centers  # [*, D, num_centers]
        rbf = torch.exp(-0.5 * (diff / sigma) ** 2)

        # if softmax:
        #     # Apply LN across the last dimension (num_centers)
        #     # rbf = self.ln(rbf)
        #     rbf=F.softmax(rbf,dim=-1)

        # Flatten last two dims: each feature D expands to D*num_centers
        rbf = rbf.reshape(*prefix, D * self.num_centers)
        return rbf

class PositionalEncoding_2d(nn.Module):
    def __init__(self, num_freqs=10):
        """
        num_freqs: number of frequency bands for Fourier features
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.freq_bands = 2.0 ** torch.arange(num_freqs) * torch.pi

    def forward(self, x):
        """
        x: [b, in_dim, h, w]
        returns: [b, in_dim * (2*num_freqs+1), h, w]
        """
        b, in_dim, h, w = x.shape
        # [b, in_dim, h, w] -> [b, h, w, in_dim]
        x_perm = x.permute(0, 2, 3, 1)

        out = [x_perm]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x_perm))
            out.append(torch.cos(freq * x_perm))

        encoded = torch.cat(out, dim=-1)  # concat on channel-like axis
        # [b, h, w, in_dim*(2*num_freqs+1)] -> [b, in_dim*(2*num_freqs+1), h, w]
        encoded = encoded.permute(0, 3, 1, 2).contiguous()
        return encoded

class PositionalEncoding_1d(nn.Module):
    def __init__(self, num_freqs=10):
        """
        num_freqs: number of frequency bands for Fourier features
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.freq_bands = 2.0 ** torch.arange(num_freqs) * torch.pi

    def forward(self, x):
        """
        x: [..., dim]  (any number of leading dimensions, last dim = coordinate/features)
        returns: [..., dim * (2*num_freqs + 1)]
        """
        orig_shape = x.shape
        dim = x.shape[-1]

        # Flatten leading dimensions
        x_flat = x.reshape(-1, dim)  # [*, dim]

        out = [x_flat]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x_flat))
            out.append(torch.cos(freq * x_flat))

        encoded = torch.cat(out, dim=-1)
        # Restore leading dimensions
        final_shape = orig_shape[:-1] + (dim * (2 * self.num_freqs + 1),)
        return encoded.view(final_shape)


class LearnableBins(nn.Module):
    def __init__(self, min_val, max_val, N):
        super().__init__()
        # initialize N+1 positive widths
        widths = torch.ones(N)
        self.width_params = nn.Parameter(widths)

        self.min_val = nn.Parameter(torch.tensor(min_val, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.max_val = nn.Parameter(torch.tensor(max_val, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.N = N

        self.ini_bin_size = (max_val - min_val) / N

    def forward(self):
        # ensure positivity
        widths = F.softplus(self.width_params)
        # normalize total width to span desired range
        widths = widths / widths.sum() * (self.max_val - self.min_val)
        # convert widths to sorted bin edges
        bin_edges = self.min_val + torch.cumsum(widths, dim=0)
        return bin_edges

class Sparsemax(nn.Module):
    def __init__(self, dim=-1):
        super(Sparsemax, self).__init__()
        self.dim = dim

    def forward(self, input):
        # Sort input along the target dimension
        input_sorted, _ = torch.sort(input, descending=True, dim=self.dim)
        input_cumsum = input_sorted.cumsum(dim=self.dim)

        # Create range values with correct broadcasting
        rhos = torch.arange(1, input.size(self.dim) + 1, device=input.device, dtype=input.dtype)
        # reshape for broadcasting
        view_shape = [1] * input.dim()
        view_shape[self.dim] = -1
        rhos = rhos.view(view_shape)

        # Compute threshold
        support = (1 + rhos * input_sorted > input_cumsum)
        k = support.sum(dim=self.dim, keepdim=True)
        tau = (input_cumsum.gather(self.dim, k - 1) - 1) / k.to(input.dtype)

        # Apply projection
        output = torch.clamp(input - tau, min=0)
        return output
class EncodedScaler(nn.Module):
    def __init__(self, N=10,min_val=0,max_val=1):
        super().__init__()
        self.sparsemax = Sparsemax(dim=1)
        self.N = N
        self.edges = LearnableBins(min_val, max_val, self.N).to('cuda')
        self.scale = nn.Parameter(torch.tensor(0., dtype=torch.float32, device='cuda'), requires_grad=True)

    def forward(self, scaler_logits):
        scaler_p = self.sparsemax(scaler_logits )
        # scaler_p = F.softmax(scaler_logits * torch.exp(self.scale), dim=1)

        centers = self.edges().view(1, self.N, 1, 1)

        scaler = (scaler_p * centers).sum(dim=1, keepdim=True)
        return scaler
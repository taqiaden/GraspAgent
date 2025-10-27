import torch
from torch import nn


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

        # if softmax:
        #
        #     for i in range(C):
        #         print()
        #         print(x[0,i,0,0])
        #         print(rbf[0,i,:,0,0].sum())
        #
        #     rbf = F.softmax(rbf, dim=2)
        #
        #     for i in range(C):
        #         print()
        #         print(x[0,i,0,0])
        #         print(rbf[0,i,:,0,0])
        #
        #     exit()
            # Apply LayerNorm per (C, W, H)
            # rbf = rbf.permute(0, 3, 4, 1, 2).contiguous()  # [B, W, H, C, num_centers]
            # rbf = self.norm(rbf)  # normalize over num_centers
            # rbf = rbf.permute(0, 3, 4, 1, 2).contiguous()  # back to [B, C, num_centers, W, H]

        rbf = rbf.view(B, C * len(self.centers), W, H)
        return rbf

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
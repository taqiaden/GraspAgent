import torch
import torch.nn as nn
import torch.nn.functional as F


def ball_query(points, centers, radius, max_neighbors):
    """
    Simple batched ball query using torch.cdist.
    Args:
        points: (B, N, 3) - support points (where we search neighbors)
        centers: (B, M, 3) - query centers
        radius: float
        max_neighbors: int
    Returns:
        idx: (B, M, K) - indices of neighbors in points, padded with N (out-of-range idx)
        mask: (B, M, K) - bool mask where True indicates real neighbor
    """
    B, N, _ = points.shape
    _, M, _ = centers.shape
    # pairwise distance: (B, M, N)
    dists = torch.cdist(centers, points, p=2)  # (B, M, N)
    within = dists <= radius  # (B, M, N)
    # For stable sorting, set distances > radius to large value
    large = radius + 1.0
    dists_masked = dists.clone()
    dists_masked[~within] = large
    # sort along N, pick first K
    sorted_dists, sorted_idx = torch.sort(dists_masked, dim=2)
    K = max_neighbors
    idx = sorted_idx[:, :, :K]  # (B, M, K)
    # mask indicates which of these are actually <= radius
    mask = sorted_dists[:, :, :K] <= radius
    # For padded indices (where mask False), set idx to N (out-of-range sentinel)
    pad_idx = torch.full((B, M, 1), N, dtype=torch.long, device=points.device)
    idx = torch.where(mask.unsqueeze(-1).expand(-1, -1, -1) , idx, torch.full_like(idx, N))
    return idx, mask


class KPConv(nn.Module):
    """
    A straightforward PyTorch implementation of the standard (rigid) KPConv.
    Ref: Hugues Thomas et al., "KPConv: Flexible and Deformable Kernel
         Convolutions for Point Clouds", ICCV 2019.

    Notes:
      - kernel_points are in local coordinates (relative to each query center).
      - We use a Gaussian influence: w_nk = exp(-||x_n - kp_k||^2 / (2 * sigma^2))
      - For each kernel point k, aggregate features: F_k = sum_n w_nk * feat_n
      - Each kernel point k has a weight matrix W_k (in_channels -> out_channels).
      - Output for a center: out = sum_k F_k @ W_k
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_points=15,
                 radius=0.5,
                 sigma=None,
                 max_neighbors=32,
                 use_bias=True):
        """
        Args:
            in_channels: int
            out_channels: int
            kernel_points: int (K)
            radius: float (ball query radius)
            sigma: float, Gaussian std. If None -> radius * 0.3 (recommended)
            max_neighbors: int, max neighbors to consider per query
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = kernel_points
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.sigma = sigma if (sigma is not None) else (radius * 0.3)

        # Weight tensor: (K, in_channels, out_channels)
        self.weight = nn.Parameter(torch.randn(self.K, in_channels, out_channels) * (2. / (in_channels + out_channels))**0.5)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize kernel point coordinates in local space (K, 3)
        # Simple heuristic: points uniformly on a sphere scaled by radius*0.5
        self.register_buffer('kernel_points', self._init_kernel_points(self.K, self.radius * 0.5))

    @staticmethod
    def _init_kernel_points(K, scale):
        """
        Initialize K points on the unit sphere (Fibonacci sphere) then scale.
        Returns tensor (K, 3)
        """
        # Fibonacci sphere for good distribution
        pts = []
        for i in range(K):
            idx = i + 0.5
            idx=torch.tensor(idx)
            phi = torch.acos(1 - 2 * idx / K)
            theta = torch.pi * (1 + 5**0.5) * idx
            x = torch.sin(phi) * torch.cos(theta)
            y = torch.sin(phi) * torch.sin(theta)
            z = torch.cos(phi)
            pts.append([x, y, z])
        pts = torch.tensor(pts, dtype=torch.float32)
        # scale
        pts = pts * scale
        return pts  # (K,3)

    def forward(self, points, features, query_points=None):
        """
        Args:
            points: (B, N, 3) - support points (source points with features)
            features: (B, N, C_in) - per-point features corresponding to `points`
            query_points: (B, M, 3) - where to compute convolutions. If None, use points (B, N, 3)
        Returns:
            out: (B, M, out_channels)
        """
        B, N, _ = points.shape
        if query_points is None:
            query_points = points
        Bq, M, _ = query_points.shape
        assert Bq == B, "Batch dims must match"

        device = points.device

        # 1) get neighbor indices and mask: idx (B, M, Kc) where Kc = max_neighbors
        idx, mask = ball_query(points, query_points, self.radius, self.max_neighbors)
        Kc = idx.shape[2]

        # 2) gather neighbor coordinates and features
        # pad points/features with sentinel at index N so out-of-range maps to zeros
        # build padded tensors (B, N+1, ...)
        pad_point = torch.zeros(B, 1, 3, device=device, dtype=points.dtype)
        pad_feat = torch.zeros(B, 1, self.in_channels, device=device, dtype=features.dtype)
        pts_padded = torch.cat([points, pad_point], dim=1)  # (B, N+1, 3)
        feats_padded = torch.cat([features, pad_feat], dim=1)  # (B, N+1, C_in)

        # idx currently (B, M, Kc) values in [0..N] where N is padding
        idx_expand = idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # for points
        neigh_pts = torch.gather(pts_padded.unsqueeze(1).expand(-1, M, -1, -1), 2, idx_expand)  # (B, M, Kc, 3)
        idx_expand_f = idx.unsqueeze(-1).expand(-1, -1, -1, self.in_channels)
        neigh_feats = torch.gather(feats_padded.unsqueeze(1).expand(-1, M, -1, -1), 2, idx_expand_f)  # (B, M, Kc, C_in)

        # 3) relative positions: (B, M, Kc, 3)
        query_expand = query_points.unsqueeze(2).expand(-1, -1, Kc, -1)
        rel_pos = neigh_pts - query_expand  # local coordinates

        # 4) compute influence weights of each neighbor for each kernel point:
        # kernel_points: (K, 3) -> expand to (1,1,1,K,3)
        K = self.K
        kp = self.kernel_points.view(1, 1, 1, K, 3)  # (1,1,1,K,3)
        # rel_pos: (B, M, Kc, 3) -> expand (B,M,Kc,1,3)
        rel_pos_k = rel_pos.unsqueeze(3)  # (B, M, Kc, 1, 3)
        diff = rel_pos_k - kp  # (B, M, Kc, K, 3)
        sq_dist = (diff ** 2).sum(dim=-1)  # (B, M, Kc, K)
        # Gaussian influence
        w = torch.exp(-sq_dist / (2 * (self.sigma ** 2)))  # (B, M, Kc, K)

        # Important: neighbors that were padding should have zero influence
        # mask: (B, M, Kc) -> expand to (B, M, Kc, K)
        mask4 = mask.unsqueeze(-1).expand(-1, -1, -1, K)
        w = w * mask4.float()

        # 5) Weighted aggregation per kernel point:
        # neigh_feats: (B, M, Kc, C_in)
        # we want F_k = sum_n w_nk * feat_n  -> result shape (B, M, K, C_in)
        w_perm = w.permute(0, 1, 3, 2)  # (B, M, K, Kc)
        neigh_feats_perm = neigh_feats.permute(0, 1, 3, 2)  # (B, M, C_in, Kc)
        # perform weighted sum along Kc: result (B, M, K, C_in)
        Fk = torch.matmul(w_perm, neigh_feats)  # (B, M, K, C_in)
        # matmul result shapes: (B,M,K,Kc) @ (B,M,Kc,C_in) -> (B,M,K,C_in)

        # 6) linear mapping per kernel point: sum_k Fk @ W_k
        # W: (K, C_in, C_out) -> expand to (1,1,K,C_in,C_out)
        W = self.weight.view(1, 1, K, self.in_channels, self.out_channels)
        Fk_unsq = Fk.unsqueeze(-2)  # (B, M, K, 1, C_in)
        # multiply and sum over in_channels: (B,M,K,1,C_in) @ (1,1,K,C_in,C_out) -> (B,M,K,1,C_out)
        out_k = torch.matmul(Fk_unsq, W).squeeze(-2)  # (B, M, K, C_out)
        # sum over K:
        out = out_k.sum(dim=2)  # (B, M, C_out)

        if self.bias is not None:
            out = out + self.bias.view(1, 1, -1)

        return out  # (B, M, out_channels)


# -------------------------
# Example / quick test
# -------------------------
if __name__ == "__main__":
    B = 2
    N = 256
    M = 64
    C_in = 8
    C_out = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    points = torch.randn(B, N, 3, device=device)
    feats = torch.randn(B, N, C_in, device=device)
    # choose some query points (e.g., subsampled)
    query = points[:, :M, :].contiguous()

    kpconv = KPConv(in_channels=C_in, out_channels=C_out, kernel_points=15, radius=0.5, max_neighbors=32).to(device)
    out = kpconv(points, feats, query)
    print("out shape:", out.shape)  # should be (B, M, C_out)

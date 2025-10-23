import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# -----------------------
# Encoder
# -----------------------
class Encoder(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------
# Decoder
# -----------------------
class Decoder(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=128, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------
# Autoencoder = Encoder + Decoder
# -----------------------
class AutoEncoder(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, in_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

# -----------------------
# Direction-aware loss
# -----------------------
def direction_loss(pred, target):
    # normalize both to unit vectors
    pred_norm = F.normalize(pred, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    # cosine similarity loss = 1 - cos(theta)
    cos_sim = torch.sum(pred_norm * target_norm, dim=-1)
    loss = 1.0 - cos_sim
    return loss.mean()

# -----------------------
# Training loop
# -----------------------
def train_autoencoder():
    # Generate toy dataset: random 3D vectors
    raw_data = torch.randn(10000, 3)
    data = F.normalize(raw_data, dim=-1)  # normalize to unit sphere
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Model, optimizer
    model = AutoEncoder(in_dim=3, hidden_dim=128, latent_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training
    n_epochs = 30
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            x_rec, z = model(x)
            loss = direction_loss(x_rec, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.6f}")

    return model

# -----------------------
# Run training
# -----------------------
if __name__ == "__main__":
    model = train_autoencoder()

    # Test on one vector
    test_vec = torch.tensor([[1.0, 2.0, 3.0]])
    test_vec = F.normalize(test_vec, dim=-1)
    rec, z = model(test_vec)

    print("\nInput (unit vector):", test_vec)
    print("Encoded (64D):", z.shape, z[0, :5], "...")
    print("Reconstructed (dir only):", F.normalize(rec, dim=-1))

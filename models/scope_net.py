from torch import nn

class scope_net(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(6, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(True),
            nn.Linear(32, 16,bias=False),
            nn.LayerNorm([16]),
            nn.ReLU(True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to('cuda')

    def forward(self, pose ): # scope includes the x,y,z point plus the approach direction
        output = self.net(pose)
        return output
import math

import torch
from colorama import Fore
from torch import nn
import torch.nn.functional as F
from torch.nn import InstanceNorm2d
from torch.nn.utils import spectral_norm

from GraspAgent_2.utils.model_init import init_orthogonal, scaled_init, init_weights_xavier, init_weights_he_normal, \
    init_weights_xavier_normal
from models.decoders import LayerNorm2D

class ParameterizedSine(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1., dtype=torch.float32, device='cuda'), requires_grad=True)
        self.theta = nn.Parameter(torch.tensor(0., dtype=torch.float32, device='cuda'), requires_grad=True)

    def forward(self,x):
        # print(x)
        # print('alpha=',self.alpha)
        # print('theta=',self.theta)

        x=torch.sin(x+self.theta)*self.alpha
        # print(f'alpha={self.alpha}, theta={self.theta}')
        return x

class ContextGate_1d_3(nn.Module):
    def __init__(self, in_c1, in_c2,mid_c=None, out_c=1,in_c3=0, relu_negative_slope=0.,
                 activation=None,use_sin=False,normalize=False,bias=False,cyclic=False):
        super().__init__()
        if mid_c is None:
            mid_c=max(in_c1,in_c2)
            mid_c+=mid_c%2

        self.gamma = nn.Sequential(
            nn.Linear(mid_c, mid_c),
        ).to('cuda')

        self.beta = nn.Sequential(
            nn.Linear(mid_c, mid_c),
        ).to('cuda')

        self.bias = nn.Sequential(
            nn.Linear(in_c1, mid_c),
        ).to('cuda')

        self.contx_proj =nn.Sequential(
            nn.Linear(in_c1, mid_c),
            # LayerNorm2D(mid_c),
            nn.SiLU(),
            # nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')


        self.cond_proj =nn.Sequential(
            nn.Linear(in_c2, mid_c),
            # LayerNorm2D(mid_c),
            # nn.SiLU(),
            # nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')

        self.d = nn.Sequential(
            # nn.SiLU(),
            nn.Linear(mid_c + in_c3, 48),
            # nn.LayerNorm(48),
            nn.SiLU(),
            nn.Linear(48, 32),
            # nn.LayerNorm(32),
            ParameterizedSine() if cyclic else nn.SiLU(),
            nn.Linear(32, out_c)
        ).to('cuda')


        self.use_sin=use_sin

        self.use_bias=bias


    def forward(self, context, condition,additional_features=None):

        context = self.contx_proj(context)
        condition = self.cond_proj(condition)


        # condition = F.normalize(condition, p=2, dim=1, eps=1e-8)
        # condition = F.softmax(condition, dim=1)

        gamma = self.gamma(context)
        beta = self.beta(context)
        # gamma = F.normalize(gamma, p=2, dim=1, eps=1e-8)

        x = condition * gamma#+beta #if self.bias else condition * gamma
        x = F.normalize(x, p=2, dim=-1, eps=1e-8)
        x = F.softmax(x, dim=-1)*beta

        if self.use_bias:
            bias = self.bias(context)

            x=x+bias


        if additional_features is not None: x=torch.cat([x,additional_features],dim=-1)
        x = self.d(x)

        return x
class MahalanobisDistance(nn.Module):
    def __init__(self, dim=64, out_dim=None, normalize=False):
        """
        dim: input feature dimension (64)
        out_dim: projected dimension (default = dim)
        normalize: whether to L2-normalize inputs before distance
        """
        super().__init__()
        out_dim =  dim if out_dim is None else out_dim
        self.normalize = normalize

        # W defines M = W^T W
        self.W = nn.Linear(dim, out_dim, bias=False)

        # Small-gain initialization for stability
        nn.init.kaiming_normal_(self.W.weight, nonlinearity="linear")
        self.W.weight.data *= 0.5

    def forward(self, main, others):
        """
        main:   [B, 1, 64]
        others: [B, N, 64]

        returns:
            dist: [B, N]
        """
        if self.normalize:
            main = F.normalize(main, dim=-1)
            others = F.normalize(others, dim=-1)

        # Broadcast main to [B, N, 64]
        diff = main - others          # [B, N, 64]

        # Apply learned transform
        z = self.W(diff)              # [B, N, out_dim]

        # Squared Mahalanobis distance
        dist = (z * z).sum(dim=-1)    # [B, N]

        return dist

class ContextGate_1d_2(nn.Module):
    def __init__(self, in_c1, in_c2):
        super().__init__()

        # def mlp(d_in, d_mid=256, d_out=64):
        #     return nn.Sequential(
        #         nn.Linear(d_in, d_mid, bias=True),
        #         nn.LayerNorm(d_mid),
        #         nn.SiLU(),
        #         nn.Linear(d_mid, d_out, bias=True),
        #         nn.LayerNorm(d_out),
        #         nn.SiLU(),
        #         nn.Linear(d_out, d_out, bias=True),
        #
        #     )

        self.context_proj = nn.Sequential(
            nn.Linear(in_c1, 128, bias=True),
            # nn.LayerNorm(128),
            # nn.SiLU(),
            # nn.Linear(128, 128, bias=True),
            nn.SiLU(),
            nn.Linear(128, 64, bias=True),
            # nn.LayerNorm(64),
            # nn.SiLU(),

        )

        self.cond_proj = nn.Sequential(
            nn.Linear(in_c2, 128, bias=True),
            # nn.LayerNorm(128),
            nn.Softmax(dim=-1),
            nn.Linear(128, 128, bias=True),
            # nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(128, 64, bias=True),

        )
        self.cond_proj1 = nn.Sequential(
            nn.Linear(in_c2, 128, bias=True),
            # nn.SiLU(),
        )
        self.cond_proj2 = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Linear(128, 64, bias=True),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
        )

        self.dist = MahalanobisDistance(dim=64,normalize=True).to('cuda')

    def forward(self, context, condition):
        condition = self.cond_proj(condition)
        # condition = F.normalize(condition, p=2, dim=-1, eps=1e-8)
        # condition = self.cond_proj2(condition)

        context = self.context_proj(context)

        condition = F.normalize(condition, p=2, dim=-1, eps=1e-8)
        context = F.normalize(context, p=2, dim=-1, eps=1e-8)

        x=context*condition
        return x.sum(dim=-1,keepdim=True)

        # condition = F.normalize(condition, p=2, dim=-1, eps=1e-8)
        # context = F.normalize(context, p=2, dim=-1, eps=1e-8)
        x = self.dist(main=context, others=condition)
        return x

class ContextGate_1d(nn.Module):
    def __init__(self, in_c1, in_c2):
        super().__init__()

        # def mlp(d_in, d_mid=256, d_out=64):
        #     return nn.Sequential(
        #         nn.Linear(d_in, d_mid, bias=True),
        #         nn.LayerNorm(d_mid),
        #         nn.SiLU(),
        #         nn.Linear(d_mid, d_out, bias=True),
        #         nn.LayerNorm(d_out),
        #         nn.SiLU(),
        #         nn.Linear(d_out, d_out, bias=True),
        #
        #     )

        self.context_proj = nn.Sequential(
            nn.Linear(in_c1, 128, bias=True),
            # nn.LayerNorm(128),
            # nn.SiLU(),
            # nn.Linear(128, 128, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64, bias=True),
            # nn.LayerNorm(64),
            nn.LeakyReLU(0.2),

        )

        self.cond_proj = nn.Sequential(
            nn.Linear(in_c2, 128, bias=True),
            # nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64, bias=True),
            # nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            # nn.Linear(128, 64, bias=True),

        )
        # self.cond_proj1 = nn.Sequential(
        #     nn.Linear(in_c2, 128, bias=True),
        #     nn.SiLU(),
        #
        # )
        # self.cond_proj2 = nn.Sequential(
        #     nn.Linear(128, 64, bias=True),
        #     nn.LayerNorm(64),
        #     nn.SiLU(),
        # )

        self.dist = MahalanobisDistance(dim=64).to('cuda')

    def forward(self, context, condition):
        condition = self.cond_proj(condition)


        context = self.context_proj(context)
        # condition = F.normalize(condition, p=2, dim=-1, eps=1e-8)
        # context = F.normalize(context, p=2, dim=-1, eps=1e-8)
        x = self.dist(main=context, others=condition)
        return x
        x = context * condition
        # x=self.d1(x)
        # x=F.softmax(x,dim=-1)
        # x=
        # x=self.d(x)
        return x.sum(dim=-1, keepdim=True)

        gamma = self.gamma(context)
        beta = self.beta(context)
        x = gamma * condition + beta

        output = self.d(x)

        return output


class ConditionalSpatiallyDisentangledDecoder(nn.Module):
    def __init__(self, n=16, latent_dim=64, cond_dim=32):
        """
        Args:
            n: number of output channels
            latent_dim: input feature channels (64)
            cond_dim: condition feature channels (c)
        """
        super().__init__()
        self.n = n

        # Shared decoder backbone
        self.shared = nn.Sequential(
            nn.Conv2d(latent_dim + cond_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
        )

        # Condition encoder (extracts spatial condition features)
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(cond_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )

        # Per-channel spatial modulation predictors (condition-aware)
        self.gamma_nets = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256 + 128, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 1, 3, padding=1),
                nn.Sigmoid()
            ) for _ in range(n)
        ])

        self.beta_nets = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256 + 128, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 1, 3, padding=1),
                nn.Tanh()
            ) for _ in range(n)
        ])

        # Final per-channel refinement with condition
        self.refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256 + 128, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 1, 3, padding=1)
            ) for _ in range(n)
        ])

    def forward(self, x, condition):
        """
        Args:
            x: [b, 64, h, w] - input features
            condition: [b, c, h, w] - conditioning features
        Returns:
            [b, n, h, w] - output with spatial diversity
        """
        # Encode condition
        cond_feat = self.cond_encoder(condition)  # [b, 128, h, w]

        # Concatenate input and condition
        combined = torch.cat([x, condition], dim=1)  # [b, 64+c, h, w]

        # Shared features
        feat = self.shared(combined)  # [b, 256, h, w]

        # Combine with condition features for modulation
        modulation_input = torch.cat([feat, cond_feat], dim=1)  # [b, 384, h, w]

        outputs = []
        for i in range(self.n):
            # Spatial masks conditioned on both feat and condition
            gamma = self.gamma_nets[i](modulation_input)  # [b,1,h,w]
            beta = self.beta_nets[i](modulation_input)  # [b,1,h,w]

            # Spatially-adaptive modulation
            modulated = feat * gamma + beta

            # Refine with condition
            refine_input = torch.cat([modulated, cond_feat], dim=1)
            out_i = self.refine[i](refine_input)
            outputs.append(out_i)

        return torch.cat(outputs, dim=1)  # [b,n,h,w]

class ContextGate_2d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c,in_c3=0, relu_negative_slope=0.,
                 activation=None,use_sin=False,normalize=False,bias=True,cyclic=False):
        super().__init__()

        mid_c=max(in_c1,in_c2)
        mid_c+=mid_c%2

        self.gamma = nn.Sequential(
            activation,
            nn.Conv2d(in_c1, mid_c, kernel_size=1),
        ).to('cuda')

        self.beta = nn.Sequential(
            activation,
            nn.Conv2d(in_c1, mid_c, kernel_size=1),
        ).to('cuda')

        self.condition_proj =nn.Sequential(
            nn.Conv2d(in_c2, mid_c, kernel_size=1),
            LayerNorm2D(mid_c),
            activation,
            nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')

        self.d = nn.Sequential(
            activation,
            nn.Conv2d(mid_c + in_c3, 48, kernel_size=1),
            LayerNorm2D(48),
            ParameterizedSine() if cyclic else activation,
            nn.Conv2d(48, out_c, kernel_size=1)
        ).to('cuda') if normalize else  nn.Sequential(
            activation,
            nn.Conv2d(mid_c + in_c3, 48, kernel_size=1),
            # LayerNorm2D(48),
            ParameterizedSine() if cyclic else activation,
            nn.Conv2d(48, out_c, kernel_size=1)
        ).to('cuda')

        self.use_sin=use_sin

        self.bias=bias


    def forward(self, context, condition,additional_features=None):
        condition = self.condition_proj(condition)

        gamma = self.gamma(context)
        beta = self.beta(context)
        # gamma = F.normalize(gamma, p=2, dim=1, eps=1e-8)

        x = condition * gamma+beta #if self.bias else condition * gamma
        # x = F.softmax(x, dim=1)*beta+bias

        if additional_features is not None: x=torch.cat([x,additional_features],dim=1)
        x = self.d(x)

        return x
class ContextGate_2d_2(nn.Module):
    def __init__(self, in_c1, in_c2, out_c,in_c3=0, relu_negative_slope=0.,
                 activation=None,use_sin=False,normalize=False,bias=True,cyclic=False):
        super().__init__()

        mid_c=max(in_c1,in_c2)
        mid_c+=mid_c%2

        self.gamma = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')

        self.beta = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')

        self.bias = nn.Sequential(
            nn.Conv2d(in_c1, mid_c, kernel_size=1),
        ).to('cuda')
        self.contx_proj = nn.Sequential(
            nn.Conv2d(in_c1, mid_c, kernel_size=1),
            # LayerNorm2D(in_c1),
            nn.SiLU(),
        ).to('cuda')

        self.cond_proj =nn.Sequential(
            nn.Conv2d(in_c2, mid_c, kernel_size=1),
            # LayerNorm2D(mid_c),
            # nn.Softmax(dim=1),
            # nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')

        self.d = nn.Sequential(
            # nn.SiLU(),
            nn.Conv2d(mid_c + in_c3, 48, kernel_size=1),
            # LayerNorm2D(48),
            nn.SiLU(),
            nn.Conv2d(48, 32, kernel_size=1),
            # LayerNorm2D(32),
            ParameterizedSine() if cyclic else nn.SiLU(),
            nn.Conv2d(32, out_c, kernel_size=1)
        ).to('cuda')

        self.use_sin=use_sin

        self.use_bias=bias

    def forward(self, context, condition,additional_features=None):

        # context = self.contx_proj(context)
        condition = self.cond_proj(condition)

        # condition = F.normalize(condition, p=2, dim=1, eps=1e-8)
        # condition = F.softmax(condition, dim=1)

        gamma = self.gamma(context)
        beta = self.beta(context)
        # gamma = F.normalize(gamma, p=2, dim=1, eps=1e-8)

        x = condition * gamma+beta #if self.bias else condition * gamma
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        x = F.softmax(x, dim=1)*beta

        if self.use_bias:
            bias = self.bias(context)
            x=x+bias

        if additional_features is not None: x=torch.cat([x,additional_features],dim=1)
        x = self.d(x)

        return x

class siren(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.w0=w0

    def forward(self,x):
        # print(x)
        # print('scale=',self.w0)
        x=torch.sin(x*self.w0*self.w)
        # print(x)
        return x
class res_ContextGate_2d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, in_c3=0, relu_negative_slope=0.,
                 activation=None, use_sin=False, normalize=False):
        super().__init__()

        mid_c = max(in_c2, in_c1)
        mid_c += mid_c % 2

        # ).to('cuda')
        if activation is None: activation = nn.LeakyReLU(relu_negative_slope)

        self.ln = nn.InstanceNorm2d(in_c1).to('cuda')

        self.gamma = nn.Sequential(
            # LayerNorm2D(in_c1),

            nn.Conv2d(in_c2, mid_c, kernel_size=1),
        ).to('cuda')

        self.beta = nn.Sequential(
            # LayerNorm2D(in_c1),

            nn.Conv2d(in_c2, mid_c, kernel_size=1),
        ).to('cuda')

        self.contx_proj = nn.Sequential(
            # LayerNorm2D(in_c1),
            # activation,
            nn.Conv2d(in_c1, mid_c, kernel_size=1),
            # LayerNorm2D(mid_c),
            # InstanceNorm2d(mid_c),
            activation,
            nn.Conv2d(mid_c, mid_c, kernel_size=1),
            # LayerNorm2D(mid_c),
            # nn.ReLU(),
            # nn.Conv2d(mid_c, mid_c, kernel_size=1),
            # ParameterizedSine(),
        ).to('cuda')

        self.res = nn.Sequential(
            nn.Conv2d(in_c1 + in_c2, 32, kernel_size=1),
            # LayerNorm2D(32),
            activation,
            nn.Conv2d(32, 32, kernel_size=1),
            # ParameterizedSine(),
        ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.cond_proj = nn.Sequential(
            nn.Conv2d(in_c2, mid_c, kernel_size=1),
            # LayerNorm2D(mid_c),
            nn.Softmax(dim=1),
            # nn.SiLU(),
            #
            nn.Conv2d(mid_c, mid_c, kernel_size=1),
            LayerNorm2D(mid_c),
            # activation,
            # nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')
        self.d1 = nn.Sequential(
            activation,
            nn.Conv2d(mid_c, mid_c, kernel_size=1),
            # LayerNorm2D(mid_c),
            activation,

        ).to('cuda')
        self.d = nn.Sequential(
            activation,
            nn.Conv2d(mid_c + in_c3, 48, kernel_size=1),
            LayerNorm2D(48),
            # nn.InstanceNorm2d(48),
            activation,
            nn.Conv2d(48, 32, kernel_size=1),
            LayerNorm2D(32),
            # nn.ReLU(),
            # nn.InstanceNorm2d(32),

            activation,
            # nn.Dropout2d(0.5),

            nn.Conv2d(32, out_c, kernel_size=1)
        ).to('cuda')

        # self.d2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(32, out_c, kernel_size=1),
        #     # LayerNorm2D(32),
        #     # nn.ReLU(),
        #     # nn.Conv2d(32, out_c, kernel_size=1)
        # ).to('cuda')
        self.use_sin = use_sin
        self.s = 1 / (mid_c ** 0.5)

    def forward(self, context, condition, additional_features=None):

        # context=self.ln(context)

        # res = self.res(torch.cat([context, condition], dim=1))

        context = self.contx_proj(context)
        # condition = self.cond_proj(condition)

        # condition = F.normalize(condition, p=2, dim=1, eps=1e-8)

        gamma = self.gamma(condition)
        beta = self.beta(condition)

        x = gamma * context
        # x = torch.sigmoid(x)
        # x = F.normalize(x, p=2, dim=1, eps=1e-8)
        # x = torch.softmax(x, dim=1)

        x = x + beta
        # x = self.d1(x)

        # x = gamma * context + beta
        # x = torch.cat([x, res], dim=1)
        if additional_features is not None: x = torch.cat([x, additional_features], dim=1)

        output = self.d(x)

        # output = F.normalize(output, p=2, dim=1, eps=1e-8)
        # output = self.d2(output)

        return output

class att_res_conv_normalized(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None,use_sigmoid=False,bottle_neck_factor=2,normalization=None):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation
        self.use_sigmoid=use_sigmoid

        # self.activation=activation
        self.key = nn.Sequential(
            # nn.InstanceNorm2d(in_c1 ),
            activation_function,
            nn.Conv2d(in_c1, in_c1 // bottle_neck_factor, kernel_size=1),
            # nn.InstanceNorm2d(in_c1 // bottle_neck_factor),
            # activation_function,
            # nn.Conv2d(in_c1 // bottle_neck_factor, in_c1 // bottle_neck_factor, kernel_size=1),
        ).to('cuda')



        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            # nn.InstanceNorm2d(in_c1),
            activation_function,
            nn.Conv2d(in_c1, in_c1 // bottle_neck_factor, kernel_size=1),
            # nn.InstanceNorm2d(in_c1 // bottle_neck_factor),
            # activation_function,
            # nn.Conv2d(in_c1// bottle_neck_factor, in_c1 // bottle_neck_factor, kernel_size=1),
        ).to('cuda')



        # self.Q_LN = LayerNorm2D(in_c2).to('cuda')
        self.query = nn.Sequential(
            nn.Conv2d(in_c2, in_c1 // (bottle_neck_factor), kernel_size=1),
            # LayerNorm2D(in_c1 // (bottle_neck_factor)),
            nn.SiLU(),
            nn.Conv2d(in_c1 // (bottle_neck_factor), in_c1 // bottle_neck_factor, kernel_size=1),
            # LayerNorm2D(in_c1 // (bottle_neck_factor)),
            # nn.SiLU(),
            # nn.Conv2d(in_c1 // (bottle_neck_factor), in_c1 // bottle_neck_factor, kernel_size=1),

        ).to('cuda')

        self.att = nn.Sequential(
            nn.Conv2d(in_c1 // bottle_neck_factor, in_c1 // bottle_neck_factor, kernel_size=1),
            # LayerNorm2D(in_c1 // bottle_neck_factor),
            activation_function,
        ).to('cuda')

        self.res = nn.Sequential(
            nn.Conv2d(in_c1 , 32, kernel_size=1),
            # LayerNorm2D(32),
            activation_function,
        ).to('cuda')


        self.sig = nn.Sigmoid()


        self.d = nn.Sequential(
            nn.Conv2d(32+(in_c1 // bottle_neck_factor), 32, kernel_size=1),
            # LayerNorm2D(32),
            activation_function,
            nn.Conv2d(32, out_c, kernel_size=1),
            # LayerNorm2D(16),
            # nn.Dropout2d(drop_out_ratio),
            # activation_function,
            # nn.Conv2d(16, out_c, kernel_size=1),
        ).to('cuda')

        self.in_c1=in_c1

    def forward(self, key_value_input, query_input):
        # key_value_input = self.LN(key_value_input)
        # normalized_key_value=self.activation(normalized_key_value)
        # normalzied_query_input=self.Q_LN(query_input)
        # query_input=self.Q_LN(query_input)



        '''residual'''
        # inputs = torch.cat([key_value_input, query_input], dim=1)
        res = self.res(key_value_input)

        '''key value from input1'''
        # key = self.key(normalized_key_value)
        value = self.value(key_value_input)

        query = self.query(query_input)

        key = self.key(key_value_input)
        # bias=self.bias(key_value_input)

        # key=key-key.mean(dim=1,keepdim=True)
        # query=query-query.mean(dim=1,keepdim=True)

        # query = F.normalize(query, p=2, dim=1, eps=1e-8)
        # key = F.normalize(key, p=2, dim=1, eps=1e-8)
        att_map = query * key

        # att_map=att_map.unflatten(1,(4,self.in_c1//8))
        # att_map = F.sigmoid(att_map)
        if self.use_sigmoid:
            # att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
            att_map = torch.sigmoid(att_map)
        else:
            att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
            att_map = F.softmax(att_map*self.scale,dim=1)
        # att_map = F.sigmoid(att_map*self.scale)

        # att_map=att_map.flatten(1,2)
        # att_map = F.softmax(att_map*self.scale,dim=1)

        # att_map = self.sig(att_map*self.scale)

        x = (att_map * value)#+bias
        x = self.att(x)

        x = torch.cat([x, res], dim=1)

        output = self.d(x)
        return output
class multi_film_decoder(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None,use_sigmoid=False,bottle_neck_factor=2,normalization=None):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation
        self.use_sigmoid=use_sigmoid

        # self.activation=activation
        self.context1 = nn.Sequential(
            nn.Conv2d(in_c1, in_c1 , kernel_size=1),
        ).to('cuda')

        self.context2 = nn.Sequential(
            activation_function,
            nn.Conv2d(in_c1, in_c1 , kernel_size=1),
        ).to('cuda')

        self.context3 = nn.Sequential(
            activation_function,
            nn.Conv2d(in_c1, in_c1 , kernel_size=1),
        ).to('cuda')

        self.approach_gamma = nn.Sequential(
            nn.Conv2d(3, in_c1, kernel_size=1),
        ).to('cuda')

        self.beta_gamma = nn.Sequential(
            nn.Conv2d(2, in_c1, kernel_size=1),
        ).to('cuda')

        self.fingers_gamma = nn.Sequential(
            nn.Conv2d(3, in_c1, kernel_size=1),
        ).to('cuda')

        self.transition_beta = nn.Sequential(
            nn.Conv2d(1, in_c1, kernel_size=1),
        ).to('cuda')

        self.point_beta = nn.Sequential(
            nn.Conv2d(1, in_c1, kernel_size=1),
        ).to('cuda')


        self.d = nn.Sequential(
            activation_function,
            nn.Conv2d(in_c1, 32, kernel_size=1,bias=False),
            # LayerNorm2D(32),
            activation_function,
            nn.Conv2d(32, 16, kernel_size=1,bias=False),
            # LayerNorm2D(16),
            activation_function,
            nn.Conv2d(16, out_c, kernel_size=1),
        ).to('cuda')

        self.in_c1=in_c1

    def forward(self, context, pose):

        approach=pose[:,0:3]
        beta=pose[:,3:5]
        fingers=pose[:,5:8]
        transition=pose[:,8:9]
        point=pose[:,9:]

        approach = self.approach_gamma(approach)
        beta=self.beta_gamma(beta)
        fingers=self.fingers_gamma(fingers)
        transition=self.transition_beta(transition)
        point=self.point_beta(point)


        context = self.context1(context)*approach+point
        context=self.context2(context)*beta+transition
        context=self.context3(context)*fingers

        output = self.d(context)
        return output

class Quality_Net_2d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, in_c3=0, relu_negative_slope=0.,
                 activation=None, use_sin=False, normalize=False):
        super().__init__()

        mid_c = max(in_c2, in_c1)
        mid_c += mid_c % 2

        # ).to('cuda')
        if activation is None: activation = nn.LeakyReLU(relu_negative_slope)

        self.ln = nn.InstanceNorm2d(in_c1).to('cuda')

        self.gamma = nn.Sequential(
            # LayerNorm2D(in_c1),

            nn.Conv2d(in_c2, mid_c, kernel_size=1),
        ).to('cuda')

        self.beta = nn.Sequential(
            # LayerNorm2D(in_c1),

            nn.Conv2d(in_c2, mid_c, kernel_size=1),
        ).to('cuda')

        self.contx_proj = nn.Sequential(
            # LayerNorm2D(in_c1),
            # activation,
            nn.Conv2d(in_c1, mid_c, kernel_size=1),
            # LayerNorm2D(mid_c),
            # InstanceNorm2d(mid_c),
            activation,
            nn.Conv2d(mid_c, mid_c, kernel_size=1),
            # LayerNorm2D(mid_c),
            # nn.ReLU(),
            # nn.Conv2d(mid_c, mid_c, kernel_size=1),
            # ParameterizedSine(),
        ).to('cuda')

        self.res = nn.Sequential(
            nn.Conv2d(in_c1 + in_c2, 32, kernel_size=1),
            # LayerNorm2D(32),
            activation,
            nn.Conv2d(32, 32, kernel_size=1),
            # ParameterizedSine(),
        ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.cond_proj = nn.Sequential(
            nn.Conv2d(in_c2, mid_c, kernel_size=1),
            LayerNorm2D(mid_c),
            # nn.Softmax(dim=1),
            nn.SiLU(),

            # nn.Conv2d(mid_c, mid_c, kernel_size=1),
            # LayerNorm2D(mid_c),
            # activation,
            # nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')
        self.d1 = nn.Sequential(
            activation,
            nn.Conv2d(mid_c, mid_c, kernel_size=1),
            # LayerNorm2D(mid_c),
            activation,

        ).to('cuda')
        self.d = nn.Sequential(
            activation,
            nn.Conv2d(mid_c + in_c3, 48, kernel_size=1),
            # LayerNorm2D(48),
            # nn.InstanceNorm2d(48),
            activation,
            nn.Conv2d(48, 32, kernel_size=1),
            # LayerNorm2D(32),
            # nn.ReLU(),
            # nn.InstanceNorm2d(32),

            activation,
            # nn.Dropout2d(0.5),

            nn.Conv2d(32, out_c, kernel_size=1)
        ).to('cuda')

        # self.d2 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(32, out_c, kernel_size=1),
        #     # LayerNorm2D(32),
        #     # nn.ReLU(),
        #     # nn.Conv2d(32, out_c, kernel_size=1)
        # ).to('cuda')
        self.use_sin = use_sin
        self.s = 1 / (mid_c ** 0.5)

    def forward(self, context, condition, additional_features=None):

        # context=self.ln(context)

        # res = self.res(torch.cat([context, condition], dim=1))

        context = self.contx_proj(context)
        # condition = self.cond_proj(condition)

        # condition = F.normalize(condition, p=2, dim=1, eps=1e-8)

        gamma = self.gamma(condition)
        beta = self.beta(condition)

        # condition = torch.softmax(condition, dim=1)

        x = gamma * context
        # x = torch.sigmoid(x)
        # x = F.normalize(x, p=2, dim=1, eps=1e-8)
        # x = torch.softmax(x, dim=1)

        x = x + beta
        # x = self.d1(x)

        # x = gamma * context + beta
        # x = torch.cat([x, res], dim=1)
        if additional_features is not None: x = torch.cat([x, additional_features], dim=1)

        output = self.d(x)

        # output = F.normalize(output, p=2, dim=1, eps=1e-8)
        # output = self.d2(output)

        return output

class CSDecoder_2d(nn.Module):
    def __init__(self, in_c1, in_c2):
        super().__init__()


        mid_c=max(in_c2,in_c1)
        mid_c+=mid_c%2

        self.contx_proj = nn.Sequential(
            nn.Conv2d(in_c1, 128, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(128, mid_c, kernel_size=1),
        ).to('cuda')


        self.cond_proj = nn.Sequential(
            nn.Conv2d(in_c2, 128, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(128, mid_c, kernel_size=1),
        ).to('cuda')

        self.bias = nn.Sequential(
            nn.Conv2d(in_c1+in_c2, mid_c // 2, kernel_size=1),
            LayerNorm2D(mid_c // 2),
            nn.SiLU(),
            nn.Conv2d(mid_c // 2, 16, kernel_size=1),
            LayerNorm2D(16),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=1)
        ).to('cuda')




    def forward(self, context, condition):
        bias=self.bias(torch.cat([context,condition],dim=1))

        context = self.contx_proj(context)
        condition = self.cond_proj(condition)
        condition = F.normalize(condition, p=2, dim=1, eps=1e-8)
        context = F.normalize(context, p=2, dim=1, eps=1e-8)



        x=condition*context


        return x.sum(dim=1,keepdim=True)+bias

class Grasp_ContextGate_2d(nn.Module):
    def __init__(self, in_c1, rotation_size, transition_size,fingers_size, out_c=1, relu_negative_slope=0.,
                 activation=None,use_sin=False,normalize=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation


        self.gamma = nn.Sequential(
            nn.Conv2d(rotation_size, in_c1, kernel_size=1),
        ).to('cuda')


        self.l1 = nn.Sequential(
            nn.Conv2d(in_c1, in_c1, kernel_size=1),
        ).to('cuda')

        self.beta = nn.Sequential(
            nn.Conv2d(transition_size, in_c1, kernel_size=1),
        ).to('cuda')

        self.l2 = nn.Sequential(
            activation_function,
            nn.Conv2d(in_c1, in_c1, kernel_size=1),
        ).to('cuda')

        self.gamma2 = nn.Sequential(
            nn.Conv2d(fingers_size, in_c1, kernel_size=1),
        ).to('cuda') if fingers_size>0 else None

        self.beta2 = nn.Sequential(
            nn.Conv2d(in_c1, in_c1, kernel_size=1),
        ).to('cuda')

        self.d = nn.Sequential(
            activation_function,
            nn.Conv2d(in_c1 , in_c1, kernel_size=1),
            LayerNorm2D(in_c1),
            ParameterizedSine() if use_sin else activation_function,
            nn.Conv2d(in_c1, 32, kernel_size=1),
            activation_function,
            nn.Conv2d(32, out_c, kernel_size=1)
        ).to('cuda') if normalize else nn.Sequential(
            activation_function,
            nn.Conv2d(in_c1, in_c1, kernel_size=1),
            ParameterizedSine() if use_sin else activation_function,
            nn.Conv2d(in_c1, 32, kernel_size=1),
            activation_function,
            nn.Conv2d(32, out_c, kernel_size=1)
        ).to('cuda')

        self.use_sin=use_sin



    def forward(self, context, rotation,transition,fingers):

        gamma = self.gamma(rotation) if self.gamma is not None else 0
        beta = self.beta(transition)
        gamma2 = self.gamma2(fingers) if self.gamma2 is not None else 0
        beta2 = self.beta2(context)


        x = self.l1(context)
        x=(1+gamma)*x+beta
        x = self.l2(x)*(1+gamma2)+beta2

        output = self.d(x)
        return output

class Linear_modulation_2d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c,in_c3=0, relu_negative_slope=0.,
                 activation=None,use_sin=False,normalize=False,bias=True):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        mid_c=max(in_c1,in_c2)



        self.value = nn.Sequential(
            nn.Conv2d(in_c1, mid_c, kernel_size=1),
        ).to('cuda')



        self.query = nn.Sequential(
            nn.Conv2d(in_c2, mid_c, kernel_size=1),
        ).to('cuda')


        self.d = nn.Sequential(
            activation_function,
            nn.Conv2d(mid_c + in_c3, 48, kernel_size=1),
            LayerNorm2D(48),
            ParameterizedSine() if use_sin else activation_function,
            nn.Conv2d(48, 32, kernel_size=1),
            activation_function,
            nn.Conv2d(32, out_c, kernel_size=1)
        ).to('cuda') if normalize else nn.Sequential(
            activation_function,
            nn.Conv2d(mid_c + in_c3, 48, kernel_size=1),
            ParameterizedSine() if use_sin else activation_function,
            nn.Conv2d(48, 32, kernel_size=1),
            activation_function,
            nn.Conv2d(32, out_c, kernel_size=1)
        ).to('cuda')

        self.use_sin=use_sin



    def forward(self, context, condition,additional_features=None):
        # context = self.gate(torch.cat([context, condition], dim=1)) * context

        value = self.value(context)
        query = self.query(condition)
        # query = F.normalize(query, p=2, dim=1, eps=1e-8)


        x = value*(1+query)
        if additional_features is not None: x=torch.cat([x,additional_features],dim=1)
        output = self.d(x)
        return output


class film_fusion_2d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c,mid_c=None,in_c3=0, relu_negative_slope=0.,
                 activation=None,use_sin=False,normalize=False,decode=True,gate=True,bias=True):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation
        self.in_c3=in_c3
        if mid_c is None:mid_c=max(in_c1,in_c2)
        self.value = nn.Sequential(
            nn.Conv2d(in_c1, in_c1, kernel_size=1),

            LayerNorm2D(in_c1),
            activation_function,

            nn.Conv2d(in_c1, mid_c, kernel_size=1),
        ).to('cuda')

        self.film_gen_ = nn.Sequential(
            nn.Conv2d(in_c2, mid_c , kernel_size=1),
            LayerNorm2D(mid_c),
            activation_function,
            nn.Conv2d(mid_c, mid_c*2, kernel_size=1),
        ).to('cuda') if bias else nn.Sequential(
            nn.Conv2d(in_c2, mid_c, kernel_size=1),
            LayerNorm2D(mid_c),
            activation_function,
            nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')

        self.bias=bias

        # self.condition_encoder = nn.Sequential(
        #     nn.Conv2d(in_c2, in_c2, kernel_size=1),
        #     activation_function
        # ).to('cuda')


        self.gate = nn.Sequential(
            nn.Conv2d(in_c2, mid_c, kernel_size=1),
            LayerNorm2D(mid_c),
            nn.Sigmoid()
        ).to('cuda') if gate else None

        self.act =activation_function

        if decode:
            self.d =  nn.Sequential(
                nn.Conv2d(mid_c+in_c3, mid_c, kernel_size=1),
                LayerNorm2D(mid_c),
                ParameterizedSine() if use_sin else activation_function,
                nn.Conv2d(mid_c, 48, kernel_size=1),
                LayerNorm2D(48),
                ParameterizedSine() if use_sin else activation_function,
                nn.Conv2d(48, out_c, kernel_size=1)
            ).to('cuda') if normalize else  nn.Sequential(
                nn.Conv2d(mid_c+in_c3, mid_c, kernel_size=1),
                ParameterizedSine() if use_sin else activation_function,
                nn.Conv2d(mid_c, 48, kernel_size=1),
                activation_function,
                nn.Conv2d(48, out_c, kernel_size=1)
            ).to('cuda')

            self.decode=decode




    def forward(self, context_features, condition_features,additional_features=None):
        context_features = self.value(context_features)

        # condition_features=self.condition_encoder(condition_features)
        if self.gate is not None:
            gate = self.gate(condition_features)
            context_features=context_features*gate


        if self.bias:
            gamma,beta = self.film_gen_(condition_features).chunk(2,dim=1)
            new_context = (1 + gamma) * context_features + beta
        else:
            gamma = self.film_gen_(condition_features)
            new_context = (1 + gamma) * context_features


        new_context=self.act(new_context)

        if additional_features is not None: new_context=torch.cat([new_context,additional_features],dim=1)

        output = self.d(new_context) if self.decode else new_context

        return output

class LearnableSine(nn.Module):
    """
    Sine activation with learnable frequency parameter omega_0.

    y = sin(omega_0 * x)
    """

    def __init__(self, init_omega_0=5.0):
        super().__init__()
        # omega_0 is a learnable parameter
        self.omega_0_ =  nn.Parameter(torch.tensor(init_omega_0, dtype=torch.float32, device='cuda'), requires_grad=True)

    def forward(self, x):
        return torch.sin(self.omega_0_ * x)


class film_fusion_1d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c,mid_c=None, relu_negative_slope=0.,
                 activation=None,use_sin=False,normalize=False,decode=True,with_gate=True,bias=True):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation
        if mid_c is None:mid_c=max(in_c1,in_c2)
        self.value = nn.Sequential(
            # nn.Linear(in_c1, in_c1),
            # nn.LayerNorm(in_c1),
            # activation_function,
            nn.Linear(in_c1, mid_c),
        ).to('cuda')

        # self.condition_proj_ = nn.Sequential(
        #     nn.Linear(in_c2, mid_c ),
        # ).to('cuda')

        self.film_gen_ = nn.Sequential(
            # nn.Linear(in_c2, mid_c ),
            # nn.LayerNorm(mid_c),
            # activation_function,
            nn.Linear(in_c2, mid_c*2),
        ).to('cuda') if bias else nn.Sequential(
            # nn.Linear(in_c2, mid_c),
            # nn.LayerNorm(mid_c),
            # activation_function,
            nn.Linear(in_c2, mid_c),
        ).to('cuda')

        self.bias=bias

        self.gate = nn.Sequential(
            nn.Linear(mid_c, mid_c),
            nn.Sigmoid()
        ).to('cuda') if with_gate else None

        # self.ln = nn.LayerNorm(64).to('cuda')
        # if decode:
        self.act =activation_function

        self.decode=decode

        if decode:
            self.d = nn.Sequential(
                nn.Linear(mid_c, 48),
                nn.LayerNorm(48),
                ParameterizedSine() if use_sin else activation_function,
                nn.Linear(48, 32),
                nn.LayerNorm(32),
                ParameterizedSine() if use_sin else activation_function,
                nn.Linear(32, out_c),
            ).to('cuda') if normalize else nn.Sequential(
                nn.Linear(mid_c, 48),
                ParameterizedSine() if use_sin else activation_function,
                nn.Linear(48, 32),
                ParameterizedSine() if use_sin else activation_function,
                nn.Linear(32, out_c),
            ).to('cuda')


    def forward(self, context, condition):
        context = self.value(context)
        # condition=self.condition_proj_(condition)
        # condition=F.normalize(condition,p=2,dim=-1,eps=1e-8)

        if self.gate is not None:
            gate = self.gate(context)
            context=context*gate

        if self.bias:
            gamma,beta = self.film_gen_(condition).chunk(2,dim=-1)
            new_context=(1+gamma)*context+beta
        else:
            gamma = self.film_gen_(condition)
            new_context = (1 + gamma) * context

        new_context=self.act(new_context)

        if not self.decode: return new_context

        output = self.d(new_context)
        return output



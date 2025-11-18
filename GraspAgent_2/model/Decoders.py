import math

import torch
from torch import nn
import torch.nn.functional as F
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

class att_1d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c,relu_negative_slope=0.,  activation=None,use_sin=False,normalize=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        med_c=max(in_c1,in_c2)

        self.key = nn.Sequential(
            nn.Linear(in_c1, med_c),
        ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)

        self.value = nn.Sequential(
            nn.Linear(in_c1, med_c),
        ).to('cuda')

        self.query = nn.Sequential(
            nn.Linear(in_c2, med_c),
        ).to('cuda')

        # self.gate = nn.Sequential(
        #     nn.Linear(in_c2+in_c1, in_c1),
        #     nn.Sigmoid()
        # ).to('cuda')

        self.d = nn.Sequential(
            nn.Linear(med_c, 48),
            nn.LayerNorm(48),
            ParameterizedSine() if use_sin else activation_function,
            nn.Linear(48, out_c),
            # nn.LayerNorm(48),
            # ParameterizedSine() if use_sin else activation_function,
            # nn.Linear(48, out_c),
        ).to('cuda') if normalize else nn.Sequential(
            nn.Linear(med_c, 48),
            ParameterizedSine() if use_sin else activation_function,
            nn.Linear(48, out_c),
            # ParameterizedSine() if use_sin else activation_function,
            # nn.Linear(32, out_c),
        ).to('cuda')



    def forward(self, context, condition):

        # context=self.gate(torch.cat([context,condition],dim=2))*context

        key = self.key(context)
        value = self.value(context)
        query = self.query(condition)

        att_map = query * key

        att_map = F.normalize(att_map, p=2, dim=2, eps=1e-8)
        # att_map = F.softmax(att_map*self.scale,dim=2)
        att_map = F.sigmoid(att_map)

        x = (att_map * value)

        output = self.d(x)
        return output

class att_2d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None,use_sin=False,normalize=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        mid_c=max(in_c1,in_c2)

        self.key = nn.Sequential(
            nn.Conv2d(in_c1, mid_c, kernel_size=1),
        ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            nn.Conv2d(in_c1, mid_c, kernel_size=1),
        ).to('cuda')

        self.query = nn.Sequential(
            nn.Conv2d(in_c2, mid_c, kernel_size=1),
        ).to('cuda')


        self.d = nn.Sequential(
            nn.Conv2d(mid_c, 48, kernel_size=1),
            LayerNorm2D(48),
            ParameterizedSine() if use_sin else activation_function,
            nn.Conv2d(48, 32, kernel_size=1),
            ParameterizedSine() if use_sin else activation_function,
            nn.Conv2d(32, out_c, kernel_size=1).to('cuda')

        ).to('cuda') if normalize else nn.Sequential(
            nn.Conv2d(mid_c, 48, kernel_size=1),
            ParameterizedSine() if use_sin else activation_function,
            nn.Conv2d(48, 32, kernel_size=1),
            activation_function,
            nn.Conv2d(32, out_c, kernel_size=1).to('cuda')
        ).to('cuda')

        self.initialize_()

    def initialize_(self):
        self.key.apply(init_weights_he_normal)
        self.value.apply(init_weights_he_normal)
        self.query.apply(init_weights_he_normal)
        self.d.apply(init_weights_he_normal)

    def forward(self, context, condition):
        # context = self.gate(torch.cat([context, condition], dim=1)) * context

        value = self.value(context)
        query = self.query(condition)
        key = self.key(context)

        att_map = query * key

        att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
        # att_map = F.softmax(att_map*self.scale,dim=1)
        att_map = F.sigmoid(att_map)


        x = (att_map * value)

        output = self.d(x)
        return output




class res_att_2d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None,use_sin=False,normalize=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        mid_c=max(in_c1,in_c2)

        self.key = nn.Sequential(
            nn.Conv2d(in_c1, mid_c, kernel_size=1),
        ).to('cuda')

        self.res = nn.Sequential(
            nn.Conv2d(in_c1+in_c2, mid_c, kernel_size=1),
            activation_function
        ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            nn.Conv2d(in_c1, mid_c, kernel_size=1),
        ).to('cuda')

        self.query = nn.Sequential(
            nn.Conv2d(in_c2, mid_c, kernel_size=1),
        ).to('cuda')

        self.d = nn.Sequential(
            nn.Conv2d(mid_c*2, mid_c, kernel_size=1),
            LayerNorm2D(mid_c),
            ParameterizedSine() if use_sin else activation_function,
            nn.Conv2d(mid_c, 32, kernel_size=1),
            ParameterizedSine() if use_sin else activation_function,
            nn.Conv2d(32, out_c, kernel_size=1).to('cuda')

        ).to('cuda') if normalize else nn.Sequential(
            nn.Conv2d(mid_c*2, mid_c, kernel_size=1),
            ParameterizedSine() if use_sin else activation_function,
            nn.Conv2d(mid_c, 32, kernel_size=1),
            activation_function,
            nn.Conv2d(32, out_c, kernel_size=1).to('cuda')
        ).to('cuda')

        self.initialize_()

    def initialize_(self):
        self.key.apply(init_weights_he_normal)
        self.value.apply(init_weights_he_normal)
        self.query.apply(init_weights_he_normal)
        self.d.apply(init_weights_he_normal)


    def forward(self, context, condition):
        # context = self.gate(torch.cat([context, condition], dim=1)) * context
        res=self.res(torch.cat([context,condition],dim=1))
        value = self.value(context)
        query = self.query(condition)
        key = self.key(context)

        att_map = query * key

        att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
        # att_map = F.softmax(att_map*self.scale,dim=1)
        att_map = F.sigmoid(att_map)

        x = (att_map * value)

        output = self.d(torch.cat([x,res],dim=1))
        return output

class film_fusion_2d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c,mid_c=None, relu_negative_slope=0.,
                 activation=None,use_sin=False,normalize=False,decode=True):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        if mid_c is None:mid_c=max(in_c1,in_c2)
        self.value = nn.Sequential(
            nn.Conv2d(in_c1, mid_c, kernel_size=1),
        ).to('cuda')

        self.film_gen_ = nn.Sequential(
            nn.Conv2d(in_c2, mid_c*2, kernel_size=1),
        ).to('cuda')

        self.gate = nn.Sequential(
            nn.Conv2d(in_c2, in_c1, kernel_size=1),
            nn.Sigmoid()
        ).to('cuda')

        self.act =activation_function

        if decode:
            self.d =  nn.Sequential(
                nn.Conv2d(mid_c, 48, kernel_size=1),
                LayerNorm2D(48),
                ParameterizedSine() if use_sin else activation_function,
                nn.Conv2d(48, 32, kernel_size=1),
                ParameterizedSine() if use_sin else activation_function,
                nn.Conv2d(32, out_c, kernel_size=1).to('cuda')

            ).to('cuda') if normalize else  nn.Sequential(
                nn.Conv2d(mid_c, 48, kernel_size=1),
                ParameterizedSine() if use_sin else activation_function,
                nn.Conv2d(48, 32, kernel_size=1),
                activation_function,
                nn.Conv2d(32, out_c, kernel_size=1).to('cuda')
            ).to('cuda')

            self.decode=decode




    def forward(self, context_features, condition_features):
        # context_features=self.ln(context_features)

        gate = self.gate(condition_features)

        context_features=context_features*gate

        x = self.value(context_features)

        gamma,beta = self.film_gen_(condition_features).chunk(2,dim=1)

        new_context=(1+gamma)*x+beta
        new_context=self.act(new_context)

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
            nn.Linear(in_c1, mid_c),
        ).to('cuda')

        self.film_gen_ = nn.Sequential(
            nn.Linear(in_c2, mid_c*2),
        ).to('cuda') if bias else nn.Sequential(
            nn.Linear(in_c2, mid_c),
        ).to('cuda')

        self.bias=bias

        self.gate = nn.Sequential(
            nn.Linear(in_c2, in_c1),
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
                nn.Linear(48, out_c),
                # nn.LayerNorm(48),
                # ParameterizedSine() if use_sin else activation_function,
                # nn.Linear(48, out_c),
            ).to('cuda') if normalize else nn.Sequential(
                nn.Linear(mid_c, 48),
                ParameterizedSine() if use_sin else activation_function,
                nn.Linear(48, out_c),
                # ParameterizedSine() if use_sin else activation_function,
                # nn.Linear(32, out_c),
            ).to('cuda')


    def forward(self, context, condition):
        if self.gate is not None:
            gate = self.gate(condition)
            context=context*gate

        x = self.value(context)
        if self.bias:
            gamma,beta = self.film_gen_(condition).chunk(2,dim=-1)
            new_context=(1+gamma)*x+beta
        else:
            gamma = self.film_gen_(condition)
            new_context = (1 + gamma) * x

        new_context=self.act(new_context)

        if not self.decode: return new_context

        output = self.d(new_context)
        return output



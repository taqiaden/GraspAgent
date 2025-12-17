import math

import torch
from colorama import Fore
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



class ContextGate_1d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c):
        super().__init__()

        med_c=max(in_c1,in_c2)
        med_c+=med_c%2

        self.gamma = nn.Sequential(
            nn.Linear(med_c, med_c),
        ).to('cuda')

        self.beta = nn.Sequential(
            nn.Linear(med_c, med_c),
        ).to('cuda')

        self.contx_proj = nn.Sequential(
            nn.Linear(in_c1, 128),
            nn.SiLU(),
            # nn.Linear(128, 128),
            # nn.SiLU(),
            nn.Linear(128, med_c),
        ).to('cuda')

        self.cond_proj = nn.Sequential(
            nn.Linear(in_c2, 128),
            nn.SiLU(),
            # nn.Linear(128, 128),
            # nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, med_c),
        ).to('cuda')

        self.d1 = nn.Sequential(
            # nn.SiLU(),
            nn.Linear(med_c, 48),
            # nn.LayerNorm(48),
            nn.SiLU(),
            nn.Linear(48, 32),

        ).to('cuda')
        self.d2 = nn.Sequential(

            nn.Linear(32, 16),
            # nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(16, out_c),
        ).to('cuda')

    def forward(self, context, condition):


        condition=self.cond_proj(condition)
        context=self.contx_proj(context)
        condition=F.normalize(condition,p=2,dim=-1,eps=1e-8)
        # condition=F.softmax(condition,dim=-1)
        context=F.normalize(context,p=2,dim=-1,eps=1e-8)
        # return self.d(torch.cat([condition,context],dim=-1))
        x=context*condition
        # x=self.d1(x)
        # x=F.softmax(x,dim=-1)
        # x=
        return x.sum(dim=-1,keepdim=True)

        gamma = self.gamma(context)
        beta = self.beta(context)
        x = gamma * condition+beta

        output = self.d(x)

        return output

class ContextGate_2d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c,in_c3=0, relu_negative_slope=0.,
                 activation=None,use_sin=False,normalize=False):
        super().__init__()


        mid_c=max(in_c1,in_c2)
        mid_c+=mid_c%2

        self.gamma = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')

        self.beta = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')

        self.contx_proj = nn.Sequential(
            nn.Conv2d(in_c1, mid_c, kernel_size=1),
            # LayerNorm2D(mid_c),
            nn.SiLU(),
            # nn.Conv2d(128, mid_c, kernel_size=1),
        ).to('cuda')

        self.cond_proj =nn.Sequential(
            nn.Conv2d(in_c2, mid_c, kernel_size=1),
            # LayerNorm2D(32),
            # nn.SiLU(),
            # nn.Conv2d(32, mid_c, kernel_size=1),
        ).to('cuda')

        self.d = nn.Sequential(
            # nn.SiLU(),
            nn.Conv2d(mid_c + in_c3, 48, kernel_size=1),
            # LayerNorm2D(48),
            nn.SiLU(),
            nn.Conv2d(48, 32, kernel_size=1),
            # LayerNorm2D(32),
            nn.SiLU(),
            nn.Conv2d(32, out_c, kernel_size=1)
        ).to('cuda')


        self.use_sin=use_sin



    def forward(self, context, condition,additional_features=None):

        # context = self.contx_proj(context)
        condition = self.cond_proj(condition)


        # condition = F.normalize(condition, p=2, dim=1, eps=1e-8)
        # condition = F.softmax(condition, dim=1)

        gamma = self.gamma(context)
        beta = self.beta(context)
        # gamma = F.normalize(gamma, p=2, dim=1, eps=1e-8)

        x = condition * gamma
        x = F.softmax(x, dim=1)*beta


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
    def __init__(self, in_c1, in_c2, out_c,in_c3=0, relu_negative_slope=0.,
                 activation=None,use_sin=False,normalize=False):
        super().__init__()


        mid_c=max(in_c2,in_c1)
        mid_c+=mid_c%2

        # ).to('cuda')

        self.gamma = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')

        self.beta = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')

        self.contx_proj = nn.Sequential(
            nn.Conv2d(in_c1, mid_c, kernel_size=1),
            # LayerNorm2D(mid_c),
            nn.SiLU(),
            # nn.Conv2d(mid_c, mid_c, kernel_size=1),
            # ParameterizedSine(),

        ).to('cuda')


        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.cond_proj = nn.Sequential(
            nn.Conv2d(in_c2, 128, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(128, mid_c, kernel_size=1),
        ).to('cuda')

        self.d = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(mid_c + in_c3, mid_c // 2, kernel_size=1),
            # LayerNorm2D(mid_c // 2),
            nn.SiLU(),
            nn.Conv2d(mid_c // 2, 32, kernel_size=1),
            # LayerNorm2D(32),
            nn.SiLU(),
            nn.Conv2d(32, out_c, kernel_size=1)
        ).to('cuda')
        self.use_sin=use_sin


        self.s = 1 / (mid_c ** 0.5)

    def forward(self, context, condition,additional_features=None):
        context = self.contx_proj(context)
        # context=F.normalize(context,p=2,dim=1,eps=1e-8)
        condition = self.cond_proj(condition)
        condition = F.normalize(condition, p=2, dim=1, eps=1e-8)
        # context = F.normalize(context, p=2, dim=1, eps=1e-8)

        # condition=F.softmax(condition,dim=1)

        gamma = self.gamma(context)
        beta = self.beta(context)

        x = gamma * condition+beta

        if additional_features is not None: x=torch.cat([x,additional_features],dim=1)

        output=self.d(x)


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



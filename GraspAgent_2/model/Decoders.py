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
    def __init__(self, in_c1, in_c2, out_c,relu_negative_slope=0.,  activation=None,use_sin=False,normalize=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        med_c=max(in_c1,in_c2)
        med_c+=med_c%2


        self.key = nn.Sequential(
            nn.Linear(in_c1, med_c),
            activation_function,
            nn.Linear(med_c, med_c),

            # nn.Dropout1d(0.5)
        ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)

        self.gamma = nn.Sequential(
            nn.Linear(med_c, med_c),
        ).to('cuda')
        self.beta = nn.Sequential(
            nn.Linear(med_c, med_c),
        ).to('cuda')


        # self.gate = nn.Sequential(
        #     nn.Linear(in_c1, in_c1),
        #     nn.Sigmoid()
        # ).to('cuda')

        self.query = nn.Sequential(
            nn.Linear(in_c2, med_c),
            activation_function,
            # nn.Linear(med_c, med_c),
            # activation_function,
        ).to('cuda')


        self.d = nn.Sequential(
            activation_function,
            nn.Linear(med_c, 48),
            nn.LayerNorm(48),
            activation_function,
            nn.Linear(48, 32),
            nn.LayerNorm(32),
            activation_function,
            nn.Linear(32, out_c),
        ).to('cuda') if normalize else nn.Sequential(
            activation_function,
            nn.Linear(med_c, 48),
            ParameterizedSine() if use_sin else activation_function,
            nn.Linear(48, 32),
            activation_function,
            nn.Linear(32, out_c),
        ).to('cuda')


    def forward(self, context, condition):
        # context=self.gate(context)*context


        key = self.key(context)
        condition = self.query(condition)
        # query = F.normalize(query, p=2, dim=-1, eps=1e-8)
        condition = F.softmax(condition.unflatten(-1, (2, -1)) * self.scale, dim=1).flatten(-2, -1)

        gamma = self.gamma(condition)
        beta = self.beta(condition)

        x = gamma * key+beta
        # att_map = F.normalize(att_map, p=2, dim=-1, eps=1e-8)

        # scores=torch.softmax(att_map*self.scale/10, dim=-1)
        # scores=torch.sigmoid(att_map)

        # s=scores.max(dim=-1)[0].mean().item()
        # if s>0.95:
        #     print(Fore.RED,f'Warning 2, saturated softmax : {s}',Fore.RESET)
        #     print(query.max(dim=-1)[1])
        #     print(key.max(dim=-1)[1])
        #
        # else:
        #     print(Fore.LIGHTGREEN_EX, f'Max of softmax : {s}', Fore.RESET)

        # x = (scores * value)

        output = self.d(x)

        return output

class ContextGate_2d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c,in_c3=0, relu_negative_slope=0.,
                 activation=None,use_sin=False,normalize=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        mid_c=max(in_c1,in_c2)
        mid_c+=mid_c%2

        self.key = nn.Sequential(
            # nn.Conv2d(in_c1, in_c1, kernel_size=1),
            # LayerNorm2D(in_c1),
            # activation_function,
            nn.Conv2d(in_c1, mid_c, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(mid_c, mid_c, kernel_size=1),

        ).to('cuda')


        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.gamma = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')
        self.beta = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')

        # self.bias = nn.Sequential(
        #     nn.Conv2d(in_c1, mid_c , kernel_size=1),
        #     # LayerNorm2D(mid_c //2),
        #     # activation_function,
        #     # nn.Conv2d(mid_c //2, mid_c , kernel_size=1),
        # ).to('cuda') if not use_sin else None

        self.query = nn.Sequential(
            nn.Conv2d(in_c2, mid_c, kernel_size=1),
            activation_function,

            # LayerNorm2D(mid_c),
            # activation_function,
            # nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')



        self.d = nn.Sequential(
            activation_function,
            nn.Conv2d(mid_c + in_c3, 48, kernel_size=1),
            LayerNorm2D(48),
            activation_function,
            nn.Conv2d(48, 32, kernel_size=1),
            LayerNorm2D(32),
            ParameterizedSine(),
            nn.Conv2d(32, out_c, kernel_size=1)
        ).to('cuda') if normalize else nn.Sequential(
            activation_function,
            nn.Conv2d(mid_c + in_c3, 48, kernel_size=1),
            ParameterizedSine() if use_sin else activation_function,
            nn.Conv2d(48, 32, kernel_size=1),
            ParameterizedSine() if use_sin else activation_function,
            nn.Conv2d(32, out_c, kernel_size=1)
        ).to('cuda')

        # self.d1 = nn.Sequential(
        #     nn.Conv2d(mid_c + in_c3, 48, kernel_size=1),
        #     ParameterizedSine() if use_sin else activation_function,
        # ).to('cuda')
        #
        # self.d2 = nn.Sequential(
        #     nn.Conv2d(48, 32, kernel_size=1),
        #     activation_function,
        #     nn.Conv2d(32, out_c, kernel_size=1)
        # ).to('cuda')

        self.use_sin=use_sin



    def forward(self, context, condition,additional_features=None):

        condition = self.query(condition)
        key = self.key(context)
        # bias=0. if self.bias is None else self.bias(context)
        # query = F.normalize(query, p=2, dim=1, eps=1e-8)
        # key = F.normalize(key, p=2, dim=1, eps=1e-8)
        condition = F.softmax(condition.unflatten(1, (2, -1)) * self.scale, dim=1).flatten(1, 2)

        gamma = self.gamma(condition)
        beta = self.beta(condition)

        # key=key-key.mean(dim=1,keepdim=True)
        # query=query-query.mean(dim=1,keepdim=True)
        # if self.use_sin: key = F.normalize(key, p=2, dim=1, eps=1e-8)

        x = gamma * key+beta
        # att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)

        # if self.use_sin: att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
        # att_map = F.sigmoid(att_map)
        # att_map = torch.softmax(att_map,dim=1)#+0.1*att_map

        # s=att_map.max(dim=1)[0].mean().item()
        # if s>0.95: print(Fore.RED,f'Warning 2, saturated softmax : {s}',Fore.RESET)
        # else:
        #     print(Fore.LIGHTGREEN_EX, f'Max of softmax : {s}', Fore.RESET)

        # x=x+bias
        # x = (att_map * value)+bias
        if additional_features is not None: x=torch.cat([x,additional_features],dim=1)
        x = self.d(x)#+bias
        # x = self.d2(x)
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
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        mid_c=max(in_c2,in_c1)
        mid_c+=mid_c%2
        # self.l1=nn.Sequential(
        #     nn.Conv2d(in_c1, in_c1, kernel_size=1),
        #     LayerNorm2D(in_c1),
        #     activation_function,
        # ).to('cuda')

        self.key = nn.Sequential(

            nn.Conv2d(in_c1, mid_c, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(mid_c, mid_c, kernel_size=1),

        ).to('cuda')

        # self.mid = nn.Sequential(
        #
        #     nn.Conv2d(mid_c, mid_c, kernel_size=1),
        #
        # ).to('cuda')

        self.gamma = nn.Sequential(

            nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')

        self.beta = nn.Sequential(

            nn.Conv2d(mid_c, mid_c, kernel_size=1),
        ).to('cuda')
        #
        # self.gate = nn.Sequential(
        #     nn.Conv2d(in_c1, in_c1, kernel_size=1),
        #     nn.Sigmoid()
        # ).to('cuda')

        # self.bias = nn.Sequential(
        #     nn.Conv2d(in_c1, mid_c, kernel_size=1),
        # ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.query = nn.Sequential(
            nn.Conv2d(in_c2, mid_c, kernel_size=1),
            activation_function,
            # nn.Conv2d(mid_c, mid_c, kernel_size=1),
            # activation_function,
            # LayerNorm2D(mid_c//2 ),
            # activation_function,
            # nn.Conv2d(mid_c//2, mid_c, kernel_size=1),
        ).to('cuda')

        # self.res = nn.Sequential(
        #     nn.Conv2d(in_c1+in_c2, mid_c//2, kernel_size=1),
        #     LayerNorm2D(mid_c//2),
        #     activation_function
        # ).to('cuda')

        # self.att = nn.Sequential(
        #     nn.Conv2d(mid_c, mid_c//2, kernel_size=1),
        #     LayerNorm2D(mid_c//2),
        #     activation_function
        # ).to('cuda')
        c=int(2*(mid_c//2))



        self.d = nn.Sequential(
            activation_function,
            nn.Conv2d(mid_c + in_c3, mid_c // 2, kernel_size=1),
            LayerNorm2D(mid_c // 2),
            activation_function,
            nn.Conv2d(mid_c // 2, 32, kernel_size=1),
            LayerNorm2D(32),
            activation_function,
            nn.Conv2d(32, out_c, kernel_size=1)
        ).to('cuda')
        self.use_sin=use_sin

        self.s = 1 / (mid_c ** 0.5)


    def forward(self, context, condition,additional_features=None):
        # context=self.l1(context)

        # gamma = self.gamma(context)
        condition = self.query(condition)
        key = self.key(context)
        # res=self.res(torch.cat([context,condition],dim=1))
        # query = F.normalize(query, p=2, dim=1, eps=1e-8)
        condition = F.softmax(condition.unflatten(1, (2, -1)) * self.scale, dim=1).flatten(1, 2)
        # key = F.normalize(key, p=2, dim=1, eps=1e-8)
        # key=key-key.mean(dim=1,keepdim=True)
        # query=query-query.mean(dim=1,keepdim=True)
        gamma = self.gamma(condition)
        beta = self.beta(condition)

        x = gamma * key+beta
        # att_map=self.mid(att_map)
        # att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
        # att_map = F.sigmoid(att_map)
        # print(att_map[0,:,0,0])
        # exit()
        # att_map = torch.softmax(att_map,dim=1)

        # s=att_map.max(dim=1)[0].mean().item()
        # if s>0.95: print(Fore.RED,f'Warining 3, saturated softmax : {s}',Fore.RESET)

        # x = (att_map * value)
        # x=self.att(x)

        # x=torch.cat([x,res],dim=1)

        if additional_features is not None: x=torch.cat([x,additional_features],dim=1)

        output=self.d(x)
        return output

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



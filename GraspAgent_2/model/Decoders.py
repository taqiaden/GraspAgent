import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

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
        # print(x)
        return x

class custom_att_1d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c,med_c=64, relu_negative_slope=0.,  activation=None):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        self.key = nn.Sequential(
            nn.Linear(in_c1, med_c),
        ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)

        self.value = nn.Sequential(
            nn.Linear(in_c1, med_c),
        ).to('cuda')

        # self.query = nn.Sequential(
        #     nn.Linear(in_c1, med_c),
        # ).to('cuda')

        # self.opening_mask = nn.Sequential(
        #     nn.Linear(1, med_c),
        # ).to('cuda')


        self.implicit_transformation=nn.Sequential(
            nn.Linear(7, med_c*med_c),
        ).to('cuda')

        self.d = nn.Sequential(
            nn.Linear(med_c, 64),
            activation_function,
            nn.Linear(64, 32),
            activation_function,
            nn.Linear(32, out_c),
        ).to('cuda')

        self.med_c=med_c

    def forward(self, key_value_input, query_input):

        # T=query_input[...,0:6]
        # w=query_input[...,6:]

        T=self.implicit_transformation(query_input).unflatten(dim=-1,sizes=(self.med_c,self.med_c))
        # d=self.translation(d)
        # w=self.opening_mask(w)
        # w = F.softmax(w,dim=-1)
        # w = F.sigmoid(w*self.scale)

        key = self.key(key_value_input)
        value = self.value(key_value_input)
        # query = self.query(key_value_input)

        # print(value.shape)
        # print(T.shape)

        query=T@key_value_input.unsqueeze(-1)

        query=(query.squeeze())#*w

        # print(value.shape)
        # exit()

        att_map = query * key

        att_map = F.normalize(att_map, p=2, dim=-1, eps=1e-8)
        value = F.softmax(value*self.scale,dim=-1)
        # att_map = F.sigmoid(att_map*self.scale)

        x = (att_map * value)

        output = self.d(x)
        return output

class normalize_free_att_1d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c,med_c=64, relu_negative_slope=0.,  activation=None,use_sin=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

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

        self.d = nn.Sequential(
            nn.Linear(med_c, 64),
            ParameterizedSine() if use_sin else activation_function,
            nn.Linear(64, 32),
            ParameterizedSine() if use_sin else activation_function,
            nn.Linear(32, out_c),
        ).to('cuda')

        self.in_c1=in_c1

    def forward(self, key_value_input, query_input):

        key = self.key(key_value_input)
        value = self.value(key_value_input)
        query = self.query(query_input)

        att_map = query * key

        att_map = F.normalize(att_map, p=2, dim=2, eps=1e-8)
        att_map = F.softmax(att_map*self.scale,dim=2)

        x = (att_map * value)

        output = self.d(x)
        return output

class normalize_free_att_sins(nn.Module):
    def __init__(self, in_c1, in_c2, out_c):
        super().__init__()


        self.key = nn.Sequential(
            nn.Linear(128, 64),
        ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)

        self.value = nn.Sequential(
            nn.Linear(128, 64),
        ).to('cuda')

        self.query = nn.Sequential(
            nn.Linear(in_c2, 64),
        ).to('cuda')

        self.d = nn.Sequential(
            nn.Linear(64, 64),
            ParameterizedSine(),
            nn.Linear(64, 32),
            ParameterizedSine(),
            nn.Linear(32, out_c),
        ).to('cuda')

        self.in_c1=in_c1

    def forward(self, key_value_input, query_input):
        key = self.key(key_value_input)
        value = self.value(key_value_input)
        query = self.query(query_input)

        att_map = query * key

        att_map = F.normalize(att_map, p=2, dim=2, eps=1e-8)
        att_map = F.softmax(att_map*self.scale,dim=2)

        x = (att_map * value)

        output = self.d(x)
        return output

class normalized_att_1d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c,med_c=64, relu_negative_slope=0.,
                 activation=None,use_sigmoid=False,):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation
        self.use_sigmoid=use_sigmoid

        self.key = nn.Sequential(
            nn.Linear(in_c1, med_c),
        ).to('cuda')


        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            nn.Linear(in_c1, med_c),
        ).to('cuda')

        self.Q_LN = nn.LayerNorm([in_c2]).to('cuda')
        self.query = nn.Sequential(
            nn.Linear(in_c2, med_c // 2),
            nn.SiLU(),
            nn.Linear(med_c // 2, med_c),
        ).to('cuda')

        self.att = nn.Sequential(
            nn.Linear(med_c, med_c//2),
            nn.LayerNorm(med_c//2),
            activation_function,
        ).to('cuda')

        self.res = nn.Sequential(
            nn.Linear(in_c1 + in_c2, 32, bias=False),
            nn.LayerNorm(32),
            activation_function,
        ).to('cuda')


        self.sig = nn.Sigmoid()


        self.d = nn.Sequential(
            nn.Linear(32+(med_c // 2), 64, bias=False),
            nn.LayerNorm([64]),
            activation_function,
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            activation_function,
            nn.Linear(32, out_c),
        ).to('cuda')

        self.in_c1=in_c1

    def forward(self, key_value_input, query_input):

        '''residual'''
        inputs = torch.cat([key_value_input, query_input], dim=-1)
        res = self.res(inputs)

        '''key value from input1'''
        # key = self.key(normalized_key_value)
        value = self.value(key_value_input)

        query = self.query(query_input)


        key = self.key(key_value_input)
        att_map = query * key

        att_map = F.normalize(att_map, p=2, dim=-1, eps=1e-8)
        if self.use_sigmoid:
            att_map = F.sigmoid(att_map*self.scale)
        else:
            att_map = F.softmax(att_map,dim=-1)

        x = (att_map * value)
        x = self.att(x)

        x = torch.cat([x, res], dim=-1)

        output = self.d(x)
        return output

class normalize_free_att_2d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None,use_sin=False,softmax_att=True,SN=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        self.key = nn.Sequential(
            nn.Conv2d(in_c1, 64, kernel_size=1),
        ).to('cuda')


        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            nn.Conv2d(in_c1, 64, kernel_size=1),
        ).to('cuda')

        self.query = nn.Sequential(
            nn.Conv2d(in_c2, 64, kernel_size=1),
        ).to('cuda')

        self.ln=LayerNorm2D(64).to('cuda')
        if SN:

            self.d = nn.Sequential(
                spectral_norm(nn.Conv2d(64, 64, kernel_size=1)),
                ParameterizedSine() if use_sin else activation_function,
                spectral_norm(nn.Conv2d(64, 32, kernel_size=1)),
                ParameterizedSine() if use_sin else activation_function,
                nn.Conv2d(32, out_c, kernel_size=1),
            ).to('cuda')
        else:
            self.d = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1),
                ParameterizedSine() if use_sin else activation_function,
                nn.Conv2d(64, 32, kernel_size=1),
                ParameterizedSine() if use_sin else activation_function,
                nn.Conv2d(32, out_c, kernel_size=1),
            ).to('cuda')

        self.softmax_att=softmax_att

    def forward(self, key_value_input, query_input):
        normalized_key_value_input=self.ln(key_value_input)

        value = self.value(key_value_input)
        query = self.query(query_input)
        key = self.key(normalized_key_value_input)

        att_map = query * key

        att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
        if self.softmax_att:
            att_map = F.softmax(att_map*self.scale,dim=1)
        else:
            att_map = F.sigmoid(att_map*self.scale)

        x = (att_map * value)

        output = self.d(x)
        return output

class normalized_att_2d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,
                 activation=None,use_sin=False,softmax_att=True):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        self.key = nn.Sequential(
            nn.Conv2d(in_c1, 64, kernel_size=1),
        ).to('cuda')


        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            nn.Conv2d(in_c1, 64, kernel_size=1),
        ).to('cuda')

        self.query = nn.Sequential(
            nn.Conv2d(in_c2, 64, kernel_size=1),
        ).to('cuda')

        self.ln=LayerNorm2D(64).to('cuda')

        self.d = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            LayerNorm2D(64),
            ParameterizedSine() if use_sin else activation_function,
            nn.Conv2d(64, 32, kernel_size=1),
            LayerNorm2D(32),
            ParameterizedSine() if use_sin else activation_function,
            nn.Conv2d(32, out_c, kernel_size=1),
        ).to('cuda')

        self.softmax_att=softmax_att

    def forward(self, key_value_input, query_input):
        normalized_key_value_input=self.ln(key_value_input)

        value = self.value(key_value_input)
        query = self.query(query_input)
        key = self.key(normalized_key_value_input)

        att_map = query * key

        att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
        if self.softmax_att:
            att_map = F.softmax(att_map*self.scale,dim=1)
        else:
            att_map = F.sigmoid(att_map*self.scale)

        x = (att_map * value)

        output = self.d(x)
        return output

class custom_att_2d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0., drop_out_ratio=0.0,
                 activation=None, use_sin=False, softmax_att=True, SN=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        self.key = nn.Sequential(
            LayerNorm2D(in_c1),
            nn.Conv2d(in_c1, 64, kernel_size=1),
        ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            LayerNorm2D(in_c1),
            nn.Conv2d(in_c1, 64, kernel_size=1),
        ).to('cuda')

        self.query = nn.Sequential(
            nn.Conv2d(in_c2, 64, kernel_size=1),
        ).to('cuda')

        # self.ln=LayerNorm2D(64).to('cuda')
        if SN:

            self.d = nn.Sequential(
                spectral_norm(nn.Conv2d(64, 64, kernel_size=1)),
                ParameterizedSine() if use_sin else activation_function,
                spectral_norm(nn.Conv2d(64, 32, kernel_size=1)),
                ParameterizedSine() if use_sin else activation_function,
                nn.Conv2d(32, out_c, kernel_size=1),
            ).to('cuda')
        else:
            self.d = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1),
                ParameterizedSine() if use_sin else activation_function,
                nn.Conv2d(64, 32, kernel_size=1),
                ParameterizedSine() if use_sin else activation_function,
                nn.Conv2d(32, out_c, kernel_size=1),
            ).to('cuda')

        self.softmax_att = softmax_att

    def forward(self, key_value_input, query_input):
        # key_value_input=self.ln(key_value_input)

        value = self.value(key_value_input)
        query = self.query(query_input)
        key = self.key(key_value_input)

        att_map = query * key

        att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
        if self.softmax_att:
            att_map = F.softmax(att_map * self.scale, dim=1)
        else:
            att_map = F.sigmoid(att_map * self.scale)

        x = (att_map * value)

        output = self.d(x)
        return output
        value = self.value(key_value_input)
        query = self.query(query_input)
        key = self.key(key_value_input)

        att_map = query * key

        att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
        if self.softmax_att:
            att_map = F.softmax(att_map*self.scale,dim=1)
        else:
            att_map = F.sigmoid(att_map*self.scale)

        x = (att_map * value)

        output = self.d(x)
        return output

class film_fusion_2d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,
                 activation=None,use_sin=False,normalize=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation


        self.value = nn.Sequential(
            nn.Conv2d(in_c1, 64, kernel_size=1),
        ).to('cuda')

        self.film_gen = nn.Sequential(
            nn.Conv2d(in_c2, 64*2, kernel_size=1),
            # LayerNorm2D(64),
            # nn.Conv2d(64, 64 * 2, kernel_size=1),
        ).to('cuda')

        self.gate = nn.Sequential(
            nn.Conv2d(in_c2, in_c1, kernel_size=1),
            nn.Sigmoid()
        ).to('cuda')

        self.d =  nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            LayerNorm2D(64),
            ParameterizedSine() if use_sin else activation_function,
            nn.Conv2d(64, 32, kernel_size=1),
            LayerNorm2D(32),
            ParameterizedSine() if use_sin else activation_function,
            nn.Conv2d(32, out_c, kernel_size=1),
        ).to('cuda') if normalize else  nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            # LayerNorm2D(64),
            ParameterizedSine() if use_sin else activation_function,
            nn.Conv2d(64, 32, kernel_size=1),
            # LayerNorm2D(32),
            ParameterizedSine() if use_sin else activation_function,
            nn.Conv2d(32, out_c, kernel_size=1),
        ).to('cuda')


    def forward(self, key_value_input, query_input):
        gate=self.gate(query_input)
        key_value_input=key_value_input*gate

        x = self.value(key_value_input)
        gamma,beta = self.film_gen(query_input).chunk(2,dim=1)
        # key = self.key(key_value_input)

        x=gamma*x+beta


        output = self.d(x)
        return output

class film_fusion_1d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,
                 activation=None,use_sin=False,normalize=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation


        self.value = nn.Sequential(
            nn.Linear(in_c1, 64),
            nn.LayerNorm(64),
            activation_function,
            nn.Linear(64, 64),
        ).to('cuda') if normalize else nn.Sequential(
            nn.Linear(in_c1, 64),
            activation_function,
            nn.Linear(64, 64),
        ).to('cuda')


        self.film_gen = nn.Sequential(
            nn.Linear(in_c2, 64*2),
            # LayerNorm2D(64),
            # activation_function,
            # nn.Conv2d(64, 64 * 2, kernel_size=1),
        ).to('cuda')

        self.d = nn.Sequential(
            nn.Linear(64, 64),
            ParameterizedSine() if use_sin else activation_function,
            nn.Linear(64, 32),
            ParameterizedSine() if use_sin else activation_function,
            nn.Linear(32, out_c),
        ).to('cuda')




    def forward(self, key_value_input, query_input):

        x = self.value(key_value_input)
        gamma,beta = self.film_gen(query_input).chunk(2,dim=-1)
        # key = self.key(key_value_input)

        x=gamma*x+beta


        output = self.d(x)
        return output

class normalize_free_res_att_2d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,
                 activation=None,use_sigmoid=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation
        self.use_sigmoid=use_sigmoid

        # self.activation=activation
        self.key = nn.Sequential(
            nn.Conv2d(in_c1, 64, kernel_size=1),
        ).to('cuda')


        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            nn.Conv2d(in_c1, 64, kernel_size=1),
        ).to('cuda')

        self.query = nn.Sequential(
            nn.Conv2d(in_c2, 64, kernel_size=1),
        ).to('cuda')

        self.att = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            activation_function,
        ).to('cuda')

        self.res = nn.Sequential(
            nn.Conv2d(in_c1 + in_c2, 32, kernel_size=1),
            activation_function,
        ).to('cuda')


        self.sig = nn.Sigmoid()


        self.d = nn.Sequential(
            nn.Conv2d(32+64, 64, kernel_size=1),
            activation_function,
            nn.Conv2d(64, 32, kernel_size=1),
            activation_function,
            nn.Conv2d(32, out_c, kernel_size=1),
        ).to('cuda')

        self.in_c1=in_c1

    def forward(self, key_value_input, query_input):

        '''residual'''
        inputs = torch.cat([key_value_input, query_input], dim=1)
        res = self.res(inputs)


        '''key value from input1'''
        # key = self.key(normalized_key_value)
        value = self.value(key_value_input)
        query = self.query(query_input)
        key = self.key(key_value_input)

        att_map = query * key

        att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)

        if self.use_sigmoid:
            att_map = F.sigmoid(att_map)
        else:
            att_map = F.softmax(att_map*self.scale,dim=1)

        x = (att_map * value)
        x = self.att(x)

        x = torch.cat([x, res], dim=1)

        output = self.d(x)
        return output

class normalized_res_att_2d(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None,use_sigmoid=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation
        self.use_sigmoid=use_sigmoid

        self.key = nn.Sequential(
            nn.Conv2d(in_c1, 64, kernel_size=1),
        ).to('cuda')


        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            nn.Conv2d(in_c1, 64, kernel_size=1),
        ).to('cuda')

        self.Q_LN = LayerNorm2D(in_c2).to('cuda')
        self.query = nn.Sequential(
            nn.Conv2d(in_c2,  64, kernel_size=1),
        ).to('cuda')

        self.att = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            LayerNorm2D(64),
            activation_function,
        ).to('cuda')

        self.res = nn.Sequential(
            nn.Conv2d(in_c1 + in_c2, 32, kernel_size=1, bias=False),
            LayerNorm2D(32),
            activation_function,
        ).to('cuda')


        self.sig = nn.Sigmoid()


        self.d = nn.Sequential(
            nn.Conv2d(32+64, 64, kernel_size=1,bias=False),
            LayerNorm2D(64),
            activation_function,
            nn.Conv2d(64, 32, kernel_size=1,bias=False),
            LayerNorm2D(32),
            nn.Dropout2d(drop_out_ratio),
            activation_function,
            nn.Conv2d(32, out_c, kernel_size=1),
        ).to('cuda')

        self.in_c1=in_c1

    def forward(self, key_value_input, query_input):


        '''residual'''
        inputs = torch.cat([key_value_input, query_input], dim=1)
        res = self.res(inputs)


        '''key value from input1'''
        # key = self.key(normalized_key_value)
        value = self.value(key_value_input)

        query = self.query(query_input)


        key = self.key(key_value_input)
        att_map = query * key

        att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)

        if self.use_sigmoid:
            att_map = F.sigmoid(att_map*self.scale)
        else:
            att_map = F.softmax(att_map,dim=1)


        x = (att_map * value)
        x = self.att(x)

        x = torch.cat([x, res], dim=1)

        output = self.d(x)
        return output






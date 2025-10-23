import torch
import torch.nn.functional as F
from torch import nn

from lib.cuda_utils import cuda_memory_report
from lib.custom_activations import GrowingCosineUnit, SwiGLU
from lib.models_utils import reshape_for_layer_norm
from models.resunet import batch_norm_relu
from registration import camera
from visualiztion import view_features
from torch.nn.utils import spectral_norm


class res_block(nn.Module):
    def __init__(self,in_c,medium_c,out_c,Batch_norm=True,Instance_norm=False):
        super().__init__()
        self.b1 = batch_norm_relu(in_c, Batch_norm=Batch_norm, Instance_norm=Instance_norm)

        self.c1 = nn.Conv2d(in_c, medium_c, kernel_size=1)
        self.b2 = batch_norm_relu(medium_c, Batch_norm=Batch_norm, Instance_norm=Instance_norm)

        self.res1 = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.c2 = nn.Conv2d(medium_c, out_c, kernel_size=1)
    def forward(self, input):
        r=self.res1(input)

        x=self.b1(input)
        x=self.c1(x)
        x=self.b2(x)
        x=self.c2(x)
        output=x+r
        return output

class res_block_mlp_LN(nn.Module):
    def __init__(self,in_c,medium_c,out_c,drop_out_ratio=0.0,relu_negative_slope=0.):
        super().__init__()
        self.ln1=nn.LayerNorm([in_c])

        self.c1 = nn.Linear(in_c, medium_c)

        self.ln2=nn.LayerNorm([medium_c])
        self.res1 = nn.Linear(in_c, out_c)
        self.c2 = nn.Linear(medium_c, out_c)
        self.drop_out=nn.Dropout(drop_out_ratio)

        self.activation = nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope>0. else nn.ReLU()
    def forward(self, input):
        r=self.res1(input)

        x=self.ln1(input)
        x = self.activation(x)
        x=self.c1(x)
        x=self.ln2(x)
        x=self.activation(x)
        x=self.drop_out(x)
        x=self.c2(x)
        output=x+r
        return output


class att_res_mlp_BN(nn.Module):
    def __init__(self,in_c1,in_c2,out_c,relu_negative_slope=0.,shallow_decoder=False):
        super().__init__()
        self.key = nn.Sequential(
            nn.Conv2d(in_c1, 32, kernel_size=1)
        ).to('cuda')

        self.value = nn.Sequential(
            nn.Conv2d(in_c1, 32, kernel_size=1)
        ).to('cuda')

        self.query =  nn.Sequential(
            nn.Conv2d(in_c2, 32, kernel_size=1)
        ).to('cuda')

        self.res=nn.Sequential(
            nn.Conv2d(in_c1+in_c2, 32, kernel_size=1,bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU(),
        ).to('cuda')

        self.att=nn.Sequential(
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU(),
        ).to('cuda')

        self.d = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1,bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU(),
            nn.Conv2d(16, out_c, kernel_size=1),
        ).to('cuda') if shallow_decoder else nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1,bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1,bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU(),
            nn.Conv2d(16, out_c, kernel_size=1),
        ).to('cuda')

    def forward(self, key_value_input,query_input):
        '''residual'''
        inputs=torch.cat([key_value_input,query_input],dim=1)
        res = self.res(inputs)

        '''key value from input1'''
        key=self.key(key_value_input)
        value=self.value(key_value_input)
        '''Query from input2'''
        query=self.query(query_input)
        '''attention score'''
        att_map=key*query
        att_map=F.softmax(att_map,dim=1)
        x=(att_map*value)
        x=self.att(x)
        x=torch.cat([x,res],dim=1)

        output=self.d(x)
        return output

class att_linear(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0., drop_out_ratio=0.0,
                  activation=None):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation



        self.key = nn.Sequential(
            nn.Linear(in_c1, 64),
        ).to('cuda')
        self.ln = nn.LayerNorm([in_c1])

        self.value = nn.Sequential(
            # nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU(),

            # nn.LayerNorm([in_c1]),
            nn.Linear(in_c1, 64),
        ).to('cuda')


        self.query = nn.Sequential(
            # nn.BatchNorm1d(in_c2),
            nn.Linear(in_c2, 64)
        ).to('cuda')


        self.sig = nn.Sigmoid()

        self.d = nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.Dropout(drop_out_ratio),
            activation_function,
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            activation_function,
            nn.Linear(16, out_c),
        ).to('cuda')


        # self.res = nn.Sequential(
        #     # nn.BatchNorm1d(in_c1+in_c2),
        #     nn.Linear(in_c1+in_c2, 32, bias=False),
        #     nn.LayerNorm([32]),
        #     activation_function,
        #     nn.Linear(32, 48),
        # ).to('cuda')
        # # self.scale = nn.Parameter(torch.tensor(0.3, dtype=torch.float32, device='cuda'), requires_grad=True)

    def forward(self, key_value_input, query_input):
        # normalized_query_input=self.ln(query_input.t()).t()
        normalized_query_input = self.ln(key_value_input)

        # res=self.res(torch.cat([key_value_input,query_input],dim=1))

        '''key value from input1'''
        key = self.key(normalized_query_input)
        value = self.value(normalized_query_input)
        '''Query from input2'''
        # query = reshape_for_layer_norm(query_input, camera=camera, reverse=True)
        # query=self.query_pre_IN(query)
        # query = reshape_for_layer_norm(query, camera=camera, reverse=False)
        query = self.query(query_input)

        '''attention score'''
        att_map = key * query
        # att_map=att_map/(32.**0.5)
        att_map = self.sig(att_map)

        x = (att_map * value)

        output = self.d(x)

        # skip_features=torch.cat([output,key_value_input,query_input],dim=1)
        # skip_features = reshape_for_layer_norm(skip_features, camera=camera, reverse=True)
        # skip_features=self.pre_IN(skip_features)
        # skip_features = reshape_for_layer_norm(skip_features, camera=camera, reverse=False)

        # delta=self.delta_(torch.cat([output,key_value_input,query_input],dim=1))
        # print(self.scale)
        # exit()

        # output=(1-self.scale)*output+self.scale*delta

        return output

class att_linear_res(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0., shallow_decoder=False, drop_out_ratio=0.0,
                 use_sigmoid=True, activation=None, normnalize_features=True):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        self.normnalize_features = normnalize_features

        self.key = nn.Sequential(
            # nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU(),
            # nn.LayerNorm([in_c1]),
            nn.Linear(in_c1, 32),
        ).to('cuda')
        self.ln = nn.LayerNorm([in_c1])
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)

        self.value = nn.Sequential(
            # nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU(),

            # nn.LayerNorm([in_c1]),
            nn.Linear(in_c1, 32),
        ).to('cuda')


        self.query = nn.Sequential(
            # nn.BatchNorm1d(in_c2),
            nn.Linear(in_c2, 32)
        ).to('cuda')

        self.res = nn.Sequential(
            nn.Linear(in_c1 + in_c2, 32, bias=False),
            nn.LayerNorm([32]),
            activation_function,
        ).to('cuda')

        self.att = nn.Sequential(
            nn.LayerNorm([32]),
            activation_function,
        ).to('cuda')
        self.sig = nn.Sigmoid()
        self.use_sig = use_sigmoid

        self.d = nn.Sequential(
            nn.Linear(64, 16, bias=False),
            nn.LayerNorm([16]),
            nn.Dropout(drop_out_ratio),
            activation_function,
            nn.Linear(16, out_c),
        ).to('cuda') if shallow_decoder else nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            activation_function,
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.Dropout(drop_out_ratio),
            activation_function,
            nn.Linear(16, out_c),
        ).to('cuda')

    def forward(self, key_value_input, query_input):
        key_value_input = reshape_for_layer_norm(key_value_input, camera=camera, reverse=False)
        query_input = reshape_for_layer_norm(query_input, camera=camera, reverse=False)
        # normalized_query_input=self.ln(query_input.t()).t()
        normalized_key_value = self.ln(key_value_input) if self.normnalize_features else key_value_input
        '''residual'''
        inputs = torch.cat([key_value_input, query_input], dim=-1)
        res = self.res(inputs)
        # print('------------------------')
        # cuda_memory_report()

        '''key value from input1'''
        # key = self.key(normalized_key_value)
        value = self.value(normalized_key_value)
        '''Query from input2'''
        # query = reshape_for_layer_norm(query_input, camera=camera, reverse=True)
        # query = self.query_pre_IN(query)
        # query = reshape_for_layer_norm(query, camera=camera, reverse=False)
        query = self.query(query_input)

        # cuda_memory_report()

        '''attention score'''
        att_map =  query #*key
        # att_map=att_map/(32.**0.5)
        if self.use_sig:
            att_map = self.sig(att_map)
        else:
            att_map = F.softmax(att_map / self.scale, dim=-1)
        x = (att_map * value)
        x = self.att(x)
        # cuda_memory_report()

        x = torch.cat([x, res], dim=-1)

        output = self.d(x)
        # cuda_memory_report()
        output = reshape_for_layer_norm(output, camera=camera, reverse=True)

        return output

class att_res_mlp_LN_SwiGLUBBlock(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0., shallow_decoder=False, drop_out_ratio=0.0,
                 use_sigmoid=True, activation=None):
        super().__init__()


        self.key = nn.Sequential(
            # nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU(),
            # nn.LayerNorm([in_c1]),
            nn.Linear(in_c1, 32),
        ).to('cuda')
        self.ln = nn.LayerNorm([in_c1])
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)

        self.value = nn.Sequential(
            # nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU(),

            # nn.LayerNorm([in_c1]),
            nn.Linear(in_c1, 32),
        ).to('cuda')


        self.query = nn.Sequential(
            nn.LayerNorm([in_c2]),
            nn.Linear(in_c2, 32)
        ).to('cuda')

        self.res = nn.Sequential(
            nn.Linear(in_c1 + in_c2, 32, bias=False),
            nn.LayerNorm([32]),
            SwiGLU(32,32)
        ).to('cuda')

        self.att = nn.Sequential(
            nn.LayerNorm([32]),
            SwiGLU(32,32),
        ).to('cuda')
        self.sig = nn.Sigmoid()
        self.use_sig = use_sigmoid

        self.d = nn.Sequential(
            nn.Linear(64, 16, bias=False),
            nn.LayerNorm([16]),
            nn.Dropout(drop_out_ratio),
            SwiGLU(16,16),
            nn.Linear(16, out_c),
        ).to('cuda') if shallow_decoder else nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            SwiGLU(32,32),
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.Dropout(drop_out_ratio),
            SwiGLU(16,16),
            nn.Linear(16, out_c),
        ).to('cuda')

    def forward(self, key_value_input, query_input):
        # normalized_query_input=self.ln(query_input.t()).t()
        normalized_query_input = self.ln(key_value_input)
        '''residual'''
        inputs = torch.cat([key_value_input, query_input], dim=-1)
        res = self.res(inputs)
        # print('------------------------')
        # cuda_memory_report()

        '''key value from input1'''
        key = self.key(normalized_query_input)
        value = self.value(normalized_query_input)
        '''Query from input2'''
        query = self.query(query_input)

        # cuda_memory_report()

        '''attention score'''
        att_map = key * query
        # att_map=att_map/(32.**0.5)
        if self.use_sig:
            att_map = self.sig(att_map)
        else:
            att_map = F.softmax(att_map / self.scale, dim=-1)
        x = (att_map * value)
        x = self.att(x)
        # cuda_memory_report()

        x = torch.cat([x, res], dim=-1)

        output = self.d(x)
        # cuda_memory_report()

        return output

class att_res_mlp_LN_SwiGLUBBlock_sparse(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0., shallow_decoder=False, drop_out_ratio=0.0,
                 use_sigmoid=True, activation=None):
        super().__init__()


        self.key = nn.Sequential(
            # nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU(),
            # nn.LayerNorm([in_c1]),
            nn.Linear(in_c1, 32),
        ).to('cuda')
        self.ln = nn.LayerNorm([in_c1])
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)

        self.value = nn.Sequential(
            # nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU(),

            # nn.LayerNorm([in_c1]),
            nn.Linear(in_c1, 32),
        ).to('cuda')

        # self.query_IN=nn.InstanceNorm1d(in_c2)

        self.query = nn.Sequential(
            nn.LayerNorm([in_c2]),
            nn.Linear(in_c2, 32)
        ).to('cuda')


        self.att = nn.Sequential(
            nn.LayerNorm([32]),
            SwiGLU(32,32),
        ).to('cuda')
        self.sig = nn.Sigmoid()
        self.use_sig = use_sigmoid

        self.d = nn.Sequential(
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.Dropout(drop_out_ratio),
            SwiGLU(16,16),
            nn.Linear(16, out_c),
        ).to('cuda') if shallow_decoder else nn.Sequential(
            nn.Linear(32, 32, bias=False),
            nn.LayerNorm([32]),
            SwiGLU(32,32),
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.Dropout(drop_out_ratio),
            SwiGLU(16,16),
            nn.Linear(16, out_c),
        ).to('cuda')

    def forward(self, key_value_input, query_input):
        # normalized_query_input=self.ln(query_input.t()).t()
        normalized_query_input = self.ln(key_value_input)
        '''residual'''
        inputs = torch.cat([key_value_input, query_input], dim=-1)
        res = self.res(inputs)
        # print('------------------------')
        # cuda_memory_report()

        '''key value from input1'''
        key = self.key(normalized_query_input)
        value = self.value(normalized_query_input)
        '''Query from input2'''
        query = self.query(query_input)

        # cuda_memory_report()

        '''attention score'''
        att_map = key * query
        # att_map=att_map/(32.**0.5)
        if self.use_sig:
            att_map = self.sig(att_map)
        else:
            att_map = F.softmax(att_map / self.scale, dim=-1)
        x = (att_map * value)
        x = self.att(x)
        # cuda_memory_report()

        x = torch.cat([x, res], dim=-1)

        output = self.d(x)
        # cuda_memory_report()

        return output

class LayerNorm2D(nn.Module):
    def __init__(self,channels,elementwise_affine=True):
        super().__init__()
        self.norm=nn.LayerNorm([channels],elementwise_affine=elementwise_affine)

    def forward(self,x):
        x=x.permute(0,2,3,1)
        x=self.norm(x)
        x=x.permute(0,3,1,2)
        return x

class sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(w0, dtype=torch.float32, device='cuda'), requires_grad=True)

    def forward(self,x):
        # print(x)
        print('scale=',self.w0)
        x=torch.sin(x*self.w0)
        # print(x)
        return x
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
            nn.Conv2d(in_c1, in_c1 // bottle_neck_factor, kernel_size=1),
        ).to('cuda')
        # self.LN = LayerNorm2D(in_c1).to('cuda')
        # self.LN = nn.Sequential(
        #     nn.Conv2d(in_c1, 64, kernel_size=1, bias=False),
        #     LayerNorm2D(64),
        #     activation_function,
        # ).to('cuda')
        # # self.routing=nn.Sequential(
        # #     nn.Conv2d(in_c1, in_c1, kernel_size=1),
        # #     LayerNorm2D(in_c1),
        # #     nn.ReLU()
        # # ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            nn.Conv2d(in_c1, in_c1 // bottle_neck_factor, kernel_size=1),
        ).to('cuda')

        self.Q_LN = LayerNorm2D(in_c2).to('cuda')
        self.query = nn.Sequential(
            nn.Conv2d(in_c2, in_c1 // (bottle_neck_factor*2), kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_c1 // (bottle_neck_factor*2), in_c1 // bottle_neck_factor, kernel_size=1),
        ).to('cuda')

        self.att = nn.Sequential(
            nn.Conv2d(in_c1 // bottle_neck_factor, in_c1 // bottle_neck_factor, kernel_size=1),
            LayerNorm2D(in_c1 // bottle_neck_factor),
            activation_function,
        ).to('cuda')

        self.res = nn.Sequential(
            nn.Conv2d(in_c1 + in_c2, 32, kernel_size=1, bias=False),
            LayerNorm2D(32),
            activation_function,
        ).to('cuda')


        self.sig = nn.Sigmoid()


        self.d = nn.Sequential(
            nn.Conv2d(32+(in_c1 // bottle_neck_factor), 32, kernel_size=1,bias=False),
            LayerNorm2D(32),
            activation_function,
            nn.Conv2d(32, 16, kernel_size=1,bias=False),
            LayerNorm2D(16),
            nn.Dropout2d(drop_out_ratio),
            activation_function,
            nn.Conv2d(16, out_c, kernel_size=1),
        ).to('cuda')

        self.in_c1=in_c1

    def forward(self, key_value_input, query_input):
        # key_value_input = self.LN(key_value_input)
        # normalized_key_value=self.activation(normalized_key_value)
        # normalzied_query_input=self.Q_LN(query_input)
        # query_input=self.Q_LN(query_input)

        '''residual'''
        inputs = torch.cat([key_value_input, query_input], dim=1)
        res = self.res(inputs)


        '''key value from input1'''
        # key = self.key(normalized_key_value)
        value = self.value(key_value_input)

        query = self.query(query_input)


        key = self.key(key_value_input)
        att_map = query * key

        # att_map=att_map.unflatten(1,(4,self.in_c1//8))
        # att_map = F.sigmoid(att_map)
        if self.use_sigmoid:
            att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
            att_map = F.sigmoid(att_map*self.scale)
        else:
            att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
            att_map = F.softmax(att_map,dim=1)
        # att_map = F.sigmoid(att_map*self.scale)

        # att_map=att_map.flatten(1,2)
        # att_map = F.softmax(att_map*self.scale,dim=1)

        # att_map = self.sig(att_map*self.scale)

        x = (att_map * value)
        x = self.att(x)

        x = torch.cat([x, res], dim=1)

        output = self.d(x)
        return output

class att_res_conv_normalized2(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        self.key = nn.Sequential(
            nn.Conv2d(in_c1, 32, kernel_size=1,bias=False),
            LayerNorm2D(32),
            activation_function,
            nn.Conv2d(32, 32, kernel_size=1),
        ).to('cuda')


        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            nn.Conv2d(in_c1, 32, kernel_size=1,bias=False),
            LayerNorm2D(32),
            activation_function,
            nn.Conv2d(32, 32, kernel_size=1),
        ).to('cuda')

        self.query = nn.Sequential(
            nn.Conv2d(in_c2, 32, kernel_size=1,bias=False),
            LayerNorm2D(32),
            activation_function,
            nn.Conv2d(32, 32, kernel_size=1),
        ).to('cuda')

        self.att = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            LayerNorm2D(32),
            activation_function,
        ).to('cuda')

        self.res = nn.Sequential(
            nn.Conv2d(in_c1 + in_c2, 32, kernel_size=1, bias=False),
            LayerNorm2D(32),
            activation_function,
            nn.Conv2d(32, 16, kernel_size=1,bias=False),
            LayerNorm2D(16),
            activation_function,
        ).to('cuda')



        self.d = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=1,bias=False),
            LayerNorm2D(32),
            activation_function,
            nn.Conv2d(32, out_c, kernel_size=1),
        ).to('cuda')

        self.in_c1=in_c1

    def forward(self, key_value_input, query_input):
        # key_value_input = self.LN(key_value_input)
        # normalized_key_value=self.activation(normalized_key_value)
        # normalzied_query_input=self.Q_LN(query_input)
        # query_input=self.Q_LN(query_input)

        '''residual'''
        inputs = torch.cat([key_value_input, query_input], dim=1)
        res = self.res(inputs)


        '''key value from input1'''
        # key = self.key(normalized_key_value)
        value = self.value(key_value_input)

        query = self.query(query_input)


        key = self.key(key_value_input)
        att_map = query * key

        att_map=att_map.unflatten(1,(4,self.in_c1//8))
        # att_map = F.sigmoid(att_map)
        att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
        # att_map = F.softmax(att_map,dim=1)
        att_map = F.sigmoid(att_map*self.scale)

        att_map=att_map.flatten(1,2)

        # att_map=att_map.flatten(1,2)
        # att_map = F.softmax(att_map*self.scale,dim=1)

        # att_map = self.sig(att_map*self.scale)

        x = (att_map * value)
        x = self.att(x)

        x = torch.cat([x, res], dim=1)

        output = self.d(x)
        return output

class att_conv_normalized2(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None,use_sin=False,normalization=None):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        self.key = nn.Sequential(
            # nn.Conv2d(in_c1, 64, kernel_size=1,bias=False),
            # LayerNorm2D(64),
            # activation_function,
            nn.Conv2d(64, 64, kernel_size=1,bias=False),
        ).to('cuda')

        normalization=normalization if normalization is not None else LayerNorm2D
        # self.gate = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.Sigmoid()
        # ).to('cuda')

        self.ln = nn.Sequential(
            nn.Conv2d(in_c1, 64, kernel_size=1, bias=False),
            normalization(64),
            activation_function,
        ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            # nn.Conv2d(in_c1, 64, kernel_size=1,bias=False),
            # LayerNorm2D(64),
            # activation_function,
            nn.Conv2d(64, 64, kernel_size=1),
        ).to('cuda')

        self.query = nn.Sequential(
            # nn.Conv2d(in_c2, 32, kernel_size=1,bias=False),
            # LayerNorm2D(32,elementwise_affine=False),
            # activation_function,
            nn.Conv2d(in_c2, 64, kernel_size=1,bias=False),
        ).to('cuda')


        self.d = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1,bias=False),
            nn.Dropout2d(0) if use_sin else normalization(64),
            sine(10) if use_sin else activation_function,
            nn.Conv2d(64, 32, kernel_size=1,bias=False),
            nn.Dropout2d(0) if use_sin else normalization(32),
            sine(5) if use_sin else activation_function,
            nn.Conv2d(32, out_c, kernel_size=1),
        ).to('cuda')

        self.in_c1=in_c1

    def forward(self, key_value_input, query_input):
        # key_value_input = self.LN(key_value_input)
        # normalized_key_value=self.activation(normalized_key_value)
        # normalzied_query_input=self.Q_LN(query_input)
        # query_input=self.Q_LN(query_input)
        # key_value_input=self.ln(key_value_input)

        '''key value from input1'''
        # key = self.key(normalized_key_value)
        value = self.value(key_value_input)

        query = self.query(query_input)


        key = self.key(key_value_input)
        att_map = query * key

        # att_map=att_map.unflatten(1,(4,self.in_c1//8))
        # att_map = F.sigmoid(att_map)
        att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
        # att_map = F.softmax(att_map*self.scale,dim=1)
        att_map = F.sigmoid(att_map*self.scale)

        # att_map=att_map.flatten(1,2)
        # att_map = F.softmax(att_map*self.scale,dim=1)

        # att_map = self.sig(att_map*self.scale)

        x = (att_map * value)


        output = self.d(x)
        return output
class att_conv_normalized_free2(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        self.key = nn.Sequential(
            # nn.Conv2d(in_c1, 64, kernel_size=1,bias=False),
            # LayerNorm2D(64),
            # activation_function,
            nn.Conv2d(64, 64, kernel_size=1),
        ).to('cuda')


        # self.gate = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=1),
        #     nn.Sigmoid()
        # ).to('cuda')

        # self.ln = nn.Sequential(
        #     nn.Conv2d(in_c1, 64, kernel_size=1, bias=False),
        #     # LayerNorm2D(64),
        #     activation_function,
        # ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            # nn.Conv2d(in_c1, 64, kernel_size=1,bias=False),
            # LayerNorm2D(64),
            # activation_function,
            nn.Conv2d(64, 64, kernel_size=1),
        ).to('cuda')

        self.query = nn.Sequential(
            # LayerNorm2D(in_c2, elementwise_affine=False),
            nn.Conv2d(in_c2, 64, kernel_size=1),
        ).to('cuda')


        self.d = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            # LayerNorm2D(64),
            activation_function,
            nn.Conv2d(64, 32, kernel_size=1),
            # LayerNorm2D(32),
            activation_function,
            nn.Conv2d(32, out_c, kernel_size=1),
        ).to('cuda')

        self.in_c1=in_c1

    def forward(self, key_value_input, query_input):
        # key_value_input = self.LN(key_value_input)
        # normalized_key_value=self.activation(normalized_key_value)
        # normalzied_query_input=self.Q_LN(query_input)
        # query_input=self.Q_LN(query_input)
        # key_value_input=self.ln(key_value_input)

        '''key value from input1'''
        # key = self.key(normalized_key_value)
        value = self.value(key_value_input)

        query = self.query(query_input)


        key = self.key(key_value_input)

        att_map = query * key

        # att_map=att_map.unflatten(1,(4,self.in_c1//8))
        # att_map = F.sigmoid(att_map)
        att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
        # att_map=F.sigmoid(att_map)
        att_map = F.softmax(att_map*self.scale,dim=1)
        # att_map = F.sigmoid(att_map*self.scale)

        # att_map=att_map.flatten(1,2)
        # att_map = F.softmax(att_map*self.scale,dim=1)

        # att_map = self.sig(att_map*self.scale)

        x = (att_map * value)


        output = self.d(x)
        return output


class att_res_conv_normalized_free(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None,use_sigmoid=False,bottle_neck_factor=2):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation
        self.use_sigmoid=use_sigmoid
        # self.activation=activation
        self.key = nn.Sequential(
            nn.Conv2d(in_c1, in_c1 // bottle_neck_factor, kernel_size=1),
        ).to('cuda')
        # self.LN = LayerNorm2D(in_c1).to('cuda')
        # self.LN = nn.Sequential(
        #     nn.Conv2d(in_c1, 64, kernel_size=1, bias=False),
        #     LayerNorm2D(64),
        #     activation_function,
        # ).to('cuda')
        # # self.routing=nn.Sequential(
        # #     nn.Conv2d(in_c1, in_c1, kernel_size=1),
        # #     LayerNorm2D(in_c1),
        # #     nn.ReLU()
        # # ).to('cuda')

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            nn.Conv2d(in_c1, in_c1 // bottle_neck_factor, kernel_size=1),
        ).to('cuda')

        self.Q_LN = LayerNorm2D(in_c2).to('cuda')
        self.query = nn.Sequential(
            nn.Conv2d(in_c2, in_c1 // (bottle_neck_factor*2), kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_c1 // (bottle_neck_factor*2), in_c1 // bottle_neck_factor, kernel_size=1),
        ).to('cuda')

        self.att = nn.Sequential(
            nn.Conv2d(in_c1 // bottle_neck_factor, in_c1 // bottle_neck_factor, kernel_size=1),
            # LayerNorm2D(in_c1 // bottle_neck_factor),
            activation_function,
        ).to('cuda')

        self.res = nn.Sequential(
            nn.Conv2d(in_c1 + in_c2, 32, kernel_size=1, bias=False),
            # LayerNorm2D(32),
            activation_function,
        ).to('cuda')


        self.sig = nn.Sigmoid()


        self.d = nn.Sequential(
            nn.Conv2d(32+(in_c1 // bottle_neck_factor), 32, kernel_size=1,bias=False),
            # LayerNorm2D(32),
            activation_function,
            nn.Conv2d(32, 16, kernel_size=1,bias=False),
            # LayerNorm2D(16),
            nn.Dropout2d(drop_out_ratio),
            activation_function,
            nn.Conv2d(16, out_c, kernel_size=1),
        ).to('cuda')

        self.in_c1=in_c1

    def forward(self, key_value_input, query_input):
        # key_value_input = self.LN(key_value_input)
        # normalized_key_value=self.activation(normalized_key_value)
        # normalzied_query_input=self.Q_LN(query_input)
        # query_input=self.Q_LN(query_input)

        '''residual'''
        inputs = torch.cat([key_value_input, query_input], dim=1)
        res = self.res(inputs)


        '''key value from input1'''
        # key = self.key(normalized_key_value)
        value = self.value(key_value_input)

        query = self.query(query_input)


        key = self.key(key_value_input)
        att_map = query * key

        # att_map=att_map.unflatten(1,(4,self.in_c1//8))
        # att_map = F.sigmoid(att_map)
        if self.use_sigmoid:
            att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
            att_map = F.sigmoid(att_map*self.scale)
        else:
            att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
            att_map = F.softmax(att_map,dim=1)
        # att_map = F.sigmoid(att_map*self.scale)

        # att_map=att_map.flatten(1,2)
        # att_map = F.softmax(att_map*self.scale,dim=1)

        # att_map = self.sig(att_map*self.scale)

        x = (att_map * value)
        x = self.att(x)

        x = torch.cat([x, res], dim=1)

        output = self.d(x)
        return output
class vinala(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        # self.activation=activation

        self.key = nn.Sequential(
            nn.Conv2d(in_c1, in_c1, kernel_size=1),
        ).to('cuda')
        self.LN = LayerNorm2D(in_c1).to('cuda')

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            nn.Conv2d(in_c1, in_c1, kernel_size=1),
        ).to('cuda')

        self.Q_LN = LayerNorm2D(in_c2).to('cuda')
        self.query = nn.Sequential(
            LayerNorm2D(in_c2),
            nn.Conv2d(in_c2, in_c1, kernel_size=1,bias=False),
        ).to('cuda')


        self.sig = nn.Sigmoid()

        self.in_c1=in_c1


        self.d = nn.Sequential(
            nn.Conv2d(in_c1+in_c2, 64, kernel_size=1,bias=False),
            LayerNorm2D(64),
            activation_function,
            nn.Conv2d(64, 32, kernel_size=1,bias=False),
            LayerNorm2D(32),
            nn.Dropout2d(drop_out_ratio),
            activation_function,
            nn.Conv2d(32, out_c, kernel_size=1),
        ).to('cuda')

    def forward(self, key_value_input, query_input):
        # key_value_input = self.LN(key_value_input)
        # normalized_key_value=self.activation(normalized_key_value)

        # normalzied_query_input=self.Q_LN(query_input)
        '''residual'''
        # inputs = torch.cat([key_value_input, query_input], dim=1)

        # query_input=self.Q_LN(query_input)
        '''key value from input1'''

        x=torch.cat([key_value_input, query_input],dim=1)

        output = self.d(x)
        return output

class att_conv_normalized(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        # self.activation=activation

        self.key = nn.Sequential(
            nn.Conv2d(in_c1, in_c1, kernel_size=1),
            # LayerNorm2D(in_c1),
            # activation_function,
            # nn.Conv2d(in_c1, in_c1, kernel_size=1),
        ).to('cuda')
        self.LN = LayerNorm2D(in_c1).to('cuda')

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            nn.Conv2d(in_c1, in_c1, kernel_size=1),
            # LayerNorm2D(in_c1),
            # activation_function,
            # nn.Conv2d(in_c1, in_c1, kernel_size=1),
            ).to('cuda')

        self.Q_LN = LayerNorm2D(in_c2).to('cuda')
        self.query = nn.Sequential(
            # nn.Conv2d(in_c2, in_c1, kernel_size=1),
            nn.Conv2d(in_c2, 32, kernel_size=1,bias=False),
            LayerNorm2D(32),
            activation_function,
            nn.Conv2d(32, in_c1, kernel_size=1),
        ).to('cuda')

        self.sig = nn.Sigmoid()

        self.in_c1=in_c1

        self.d = nn.Sequential(
            nn.Conv2d(in_c1, 64, kernel_size=1,bias=False),
            LayerNorm2D(64),
            activation_function,
            nn.Conv2d(64, 32, kernel_size=1,bias=False),
            LayerNorm2D(32),
            nn.Dropout2d(drop_out_ratio),
            activation_function,
            nn.Conv2d(32, out_c, kernel_size=1),
        ).to('cuda')

    def forward(self, key_value_input, query_input):

        key_value_input=self.LN(key_value_input)

        value = self.value(key_value_input)
        query = self.query(query_input)
        key = self.key(key_value_input)

        # print(value[0,:,0,0:10])
        # print(query[0,:,0,0:10])
        # print(key[0,:,0,0:10])
        # print('-----------------')

        att_map =  query * key

        att_map=att_map.unflatten(1,(8,self.in_c1//8))
        # att_map = F.sigmoid(att_map)
        att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
        # att_map = F.softmax(att_map,dim=1)
        att_map = F.sigmoid(att_map*self.scale)

        att_map=att_map.flatten(1,2)


        x = (att_map * value)

        output = self.d(x)
        return output
class att_conv_normalized128(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        self.key = nn.Sequential(
            nn.Conv2d(in_c1, 64, kernel_size=1,bias=False),
            LayerNorm2D(64),
            activation_function,
            nn.Conv2d(64, 128, kernel_size=1),
        ).to('cuda')


        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            nn.Conv2d(in_c1, 64, kernel_size=1,bias=False),
            LayerNorm2D(64),
            activation_function,
            nn.Conv2d(64, 128, kernel_size=1),
        ).to('cuda')

        self.query = nn.Sequential(
            nn.Conv2d(in_c2, 32, kernel_size=1,bias=False),
            LayerNorm2D(32),
            activation_function,
            nn.Conv2d(32, 128, kernel_size=1),
        ).to('cuda')


        self.d = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1,bias=False),
            LayerNorm2D(64),
            activation_function,
            nn.Conv2d(64, 32, kernel_size=1,bias=False),
            LayerNorm2D(32),
            activation_function,
            nn.Conv2d(32, out_c, kernel_size=1),
        ).to('cuda')

        self.in_c1=in_c1

    def forward(self, key_value_input, query_input):
        # key_value_input = self.LN(key_value_input)
        # normalized_key_value=self.activation(normalized_key_value)
        # normalzied_query_input=self.Q_LN(query_input)
        # query_input=self.Q_LN(query_input)


        '''key value from input1'''
        # key = self.key(normalized_key_value)
        value = self.value(key_value_input)

        query = self.query(query_input)


        key = self.key(key_value_input)
        att_map = query * key

        # att_map=att_map.unflatten(1,(4,self.in_c1//8))
        # att_map = F.sigmoid(att_map)
        att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
        att_map = F.softmax(att_map*self.scale,dim=1)
        # att_map = F.sigmoid(att_map*self.scale)

        # att_map=att_map.flatten(1,2)
        # att_map = F.softmax(att_map*self.scale,dim=1)

        # att_map = self.sig(att_map*self.scale)

        x = (att_map * value)


        output = self.d(x)
        return output

class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / (x.pow(2).mean(dim=1, keepdim=True) + self.eps).sqrt()
class film_conv_normalized(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        # self.activation=activation

        # self.key = nn.Sequential(
        #     nn.Conv2d(in_c1, in_c1, kernel_size=1),
        # ).to('cuda')
        # self.LN = LayerNorm2D(in_c1).to('cuda')

        # self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            nn.Conv2d(in_c1, in_c1, kernel_size=1),
        ).to('cuda')

        # self.Q_LN = LayerNorm2D(in_c2).to('cuda')


        self.scale = nn.Sequential(
            # LayerNorm2D(in_c2),

            nn.Conv2d(in_c2, in_c1, kernel_size=1,bias=False),
        ).to('cuda')
        self.shift = nn.Sequential(
            # LayerNorm2D(in_c2),

            nn.Conv2d(in_c2, in_c1, kernel_size=1),
        ).to('cuda')

        self.sig = nn.Sigmoid()

        self.d = nn.Sequential(
            nn.Conv2d(in_c1, 64, kernel_size=1,bias=False),
            LayerNorm2D(64,elementwise_affine=True),
            # nn.Dropout2d(0.),
            activation_function,
            nn.Conv2d(64, 32, kernel_size=1,bias=False),
            LayerNorm2D(32,elementwise_affine=True),
            # nn.Dropout2d(0.),
            activation_function,
            nn.Conv2d(32, out_c, kernel_size=1),
        ).to('cuda')

    def forward(self, key_value_input, query_input):
        # key_value_input = self.LN(key_value_input)
        # normalized_key_value=self.activation(normalized_key_value)

        # normalzied_query_input=self.Q_LN(query_input)
        '''residual'''
        # inputs = torch.cat([key_value_input, query_input], dim=1)


        '''key value from input1'''

        value = self.value(key_value_input)

        # query_input=self.scale_invarience(query_input)
        scale = self.scale(query_input)
        shift = self.shift(query_input)


        # scale=scale.unflatten(1,(2,-1))
        # scale = F.softmax(scale, dim=1)
        # scale=scale.flatten(1,2)

        #
        modulation =  value * (1+scale)#+shift

        # att_map = self.sig(att_map)
        # att_map = F.softmax(att_map*self.scale,dim=1)

        # x = (att_map * value)

        output = self.d(modulation)
        return output
class film_conv_normalized_128(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        # self.activation=activation

        self.key = nn.Sequential(
            nn.Conv2d(in_c1, in_c1, kernel_size=1),
        ).to('cuda')
        self.LN = LayerNorm2D(in_c1).to('cuda')

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            nn.Conv2d(in_c1, in_c1*2, kernel_size=1),
        ).to('cuda')

        # self.Q_LN = LayerNorm2D(in_c2).to('cuda')
        self.query = nn.Sequential(
            LayerNorm2D(in_c2),
            nn.Conv2d(in_c2, in_c1*2, kernel_size=1),
        ).to('cuda')

        self.sig = nn.Sigmoid()

        self.d = nn.Sequential(
            nn.Conv2d(in_c1*2, in_c1, kernel_size=1),
            LayerNorm2D(in_c1),
            # nn.Dropout2d(0.),
            activation_function,
            nn.Conv2d(in_c1, 32, kernel_size=1),
            LayerNorm2D(32),
            # nn.Dropout2d(0.),
            activation_function,
            nn.Conv2d(32, out_c, kernel_size=1),
        ).to('cuda')

    def forward(self, key_value_input, query_input):
        # key_value_input = self.LN(key_value_input)
        # normalized_key_value=self.activation(normalized_key_value)

        # normalzied_query_input=self.Q_LN(query_input)
        '''residual'''
        # inputs = torch.cat([key_value_input, query_input], dim=1)


        '''key value from input1'''

        value = self.value(key_value_input)
        # key = self.key(key_value_input)

        scale = self.query(query_input)
        # scale = F.softmax(scale, dim=1)

        modulation =  value * scale#+shift

        # att_map = self.sig(att_map)
        # att_map = F.softmax(att_map*self.scale,dim=1)

        # x = (att_map * value)

        output = self.d(modulation)
        return output

class att_conv_normalized_free(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0., drop_out_ratio=0.0,
                 activation=None):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        # self.activation=activation

        self.key = nn.Sequential(
            nn.Conv2d(in_c1, in_c1, kernel_size=1),
        ).to('cuda')
        self.LN = LayerNorm2D(in_c1).to('cuda')

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        self.value = nn.Sequential(
            nn.Conv2d(in_c1, in_c1, kernel_size=1),
        ).to('cuda')

        self.Q_LN = LayerNorm2D(in_c2).to('cuda')
        self.query = nn.Sequential(
            nn.Conv2d(in_c2, in_c1, kernel_size=1),

            # nn.Conv2d(in_c2, 32, kernel_size=1,bias=False),
            # LayerNorm2D(32),
            # activation_function,
            # nn.Conv2d(32, in_c1, kernel_size=1),
        ).to('cuda')

        self.sig = nn.Sigmoid()

        self.in_c1 = in_c1

        self.d = nn.Sequential(
            nn.Conv2d(in_c1, 64, kernel_size=1, bias=False),
            # LayerNorm2D(64),
            activation_function,
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            # LayerNorm2D(32),
            nn.Dropout2d(drop_out_ratio),
            activation_function,
            nn.Conv2d(32, out_c, kernel_size=1),
        ).to('cuda')

    def forward(self, key_value_input, query_input):

        value = self.value(key_value_input)
        query = self.query(query_input)
        key = self.key(key_value_input)

        # print(value[0,:,0,0:10])
        # print(query[0,:,0,0:10])
        # print(key[0,:,0,0:10])
        # print('-----------------')

        att_map = query * key
        att_map = F.normalize(att_map, p=2, dim=1, eps=1e-8)
        att_map = F.softmax(att_map, dim=1)
        x = (att_map * value)

        output = self.d(x)
        return output

def add_spectral_norm_selective(model, layer_types=(nn.Conv2d, nn.Linear)):
    for name, layer in model.named_children():
        if isinstance(layer, layer_types):
            setattr(model, name, spectral_norm(layer))
        else:
            add_spectral_norm_selective(layer, layer_types)
    return model

class att_conv_LN(nn.Module):
    def  __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None,shallow=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        self.activation=activation_function
        s=int(in_c2*8)

        self.scale = nn.Parameter(torch.tensor(torch.ones(1,in_c2,1,1,1), dtype=torch.float32, device='cuda'), requires_grad=True)

        # self.neck = nn.Sequential(
        #     # LayerNorm2D(in_c1),
        #     nn.Conv2d(in_c1, 32, kernel_size=1),
        #     activation_function,
        #     # nn.Conv2d(32, 48, kernel_size=1)
        # ).to('cuda')

        self.s=s
        self.in_c2=in_c2

        self.key = nn.Sequential(
            # LayerNorm2D(in_c1),
            nn.Conv2d(in_c1, 32, kernel_size=1),
            # activation_function,
            # nn.Conv2d(32, 48, kernel_size=1)
        ).to('cuda')

        self.value = nn.Sequential(
            # LayerNorm2D(in_c1),
            nn.Conv2d(in_c1, 32, kernel_size=1),
            # activation_function,
            # nn.Conv2d(32, 48, kernel_size=1)
        ).to('cuda')

        # self.res = nn.Sequential(
        #     # LayerNorm2D(in_c1),
        #     nn.Conv2d(in_c1, 16, kernel_size=1),
        #     activation_function,
        # ).to('cuda')

        # self.att = nn.Sequential(
        #     # LayerNorm2D(in_c1),
        #     nn.Conv2d(48, 32, kernel_size=1),
        #     activation_function,
        # ).to('cuda')
        # self.LN = LayerNorm2D(in_c1).to('cuda')

        # self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        # self.value = nn.Sequential(
        #     nn.Conv2d(in_c1, 32, kernel_size=1),
        # ).to('cuda')

        self.LN = LayerNorm2D(in_c1).to('cuda')
        self.query = nn.Sequential(
            # nn.Conv2d(in_c2, s, kernel_size=1, groups=in_c2),
            nn.Conv2d(in_c2,in_c2*32, kernel_size=1,groups=in_c2),
            # # LayerNorm2D(32),
            # activation_function,
            # nn.Conv2d(in_c2*4, 64, kernel_size=1),
        ).to('cuda')

        self.d = nn.Sequential(
            # nn.Conv2d(in_c2, s, kernel_size=1, groups=in_c2),
            nn.Conv2d(in_c2 * 32, in_c2 * 12, kernel_size=1, groups=in_c2,bias=False),
            LayerNorm2D( in_c2 * 12),
            # nn.InstanceNorm2d(in_c2 * 12),
            activation_function,
            nn.Conv2d(in_c2 * 12, 16, kernel_size=1,bias=False),
            LayerNorm2D( 16),
            # nn.Dropout2d(0.),
            activation_function,
            nn.Conv2d(16, out_c, kernel_size=1),
        ).to('cuda')

        # self.sig = nn.Sigmoid()

        # self.d =nn.Sequential(
        #     nn.Conv2d(64, 16, kernel_size=1),
        #     activation_function,
        #     nn.Conv2d(16, out_c, kernel_size=1),
        # ).to('cuda') if shallow  else nn.Sequential(
        #     nn.Conv2d(64, 32, kernel_size=1),
        #     activation_function,
        #     nn.Conv2d(32, 16, kernel_size=1),
        #     nn.Dropout2d(drop_out_ratio),
        #     activation_function,
        #     nn.Conv2d(16, out_c, kernel_size=1),
        # ).to('cuda')

    def forward(self, key_value_input, query_input):
        key_value_input = self.LN(key_value_input)
        # key_value_input=self.activation(key_value_input)

        # normalzied_query_input=self.Q_LN(query_input)
        '''residual'''
        # inputs = torch.cat([key_value_input, query_input], dim=1)

        # key_value_input = self.neck(key_value_input)

        '''key value from input1'''
        key = self.key(key_value_input)
        value = self.value(key_value_input)
        query = self.query(query_input)

        # key=key.unflatten(1,(self.in_c2,8))
        # value=value.unflatten(1,(self.in_c2,8))
        key=key[:,None,...]
        value=value[:,None,...]

        query=query.unflatten(1,(self.in_c2,32))

        att_map = key * query

        # att_map=att_map.unflatten(1,(8,8))
        att_map=att_map*self.scale # B, c,8,h,w
        # att_map = att_map.unflatten(2, (4, 8))
        att_map=F.softmax(att_map,dim=2)
        # att_map = att_map.flatten(2, 3)

        '''attention score'''
        att_scores = value * att_map

        att_scores=att_scores.flatten(1,2)
        # att_map = self.sig(att_map*self.scale)
        # x=self.att(x)

        # x = torch.cat([x, res], dim=1)

        # x = (att_map * value)

        output = self.d(att_scores)
        return output
class att_conv_LN2(nn.Module):
    def  __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None,shallow=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        self.activation=activation_function
        s=int(in_c2*8)

        self.s=s
        self.in_c2=in_c2

        self.key = nn.Sequential(
            nn.Conv2d(in_c1, 64, kernel_size=1),
        ).to('cuda')

        self.res = nn.Sequential(
            # LayerNorm2D(in_c1),
            nn.Conv2d(in_c1, 64, kernel_size=1),
            # activation_function,
            # nn.Conv2d(32, 48, kernel_size=1)
        ).to('cuda')


        self.LN = LayerNorm2D(in_c1).to('cuda')
        self.query = nn.Sequential(
            # nn.Conv2d(in_c2, s, kernel_size=1, groups=in_c2),
            nn.Conv2d(in_c2,64, kernel_size=1),
            # LayerNorm2D(32),
            # activation_function,
            # nn.Conv2d(32, 64, kernel_size=1),
        ).to('cuda')

        self.d = nn.Sequential(
            # nn.Conv2d(in_c2, s, kernel_size=1, groups=in_c2),
            nn.Conv2d( 64, 32, kernel_size=1, bias=False),
            LayerNorm2D( 32),
            # nn.InstanceNorm2d(in_c2 * 12),
            activation_function,
            nn.Conv2d(32, 16, kernel_size=1,bias=False),
            LayerNorm2D( 16),
            # nn.Dropout2d(0.),
            activation_function,
            nn.Conv2d(16, out_c, kernel_size=1),
        ).to('cuda')


    def forward(self, key_value_input, query_input):
        key_value_input = self.LN(key_value_input)


        '''key value from input1'''
        key = self.key(key_value_input)
        res = self.res(key_value_input)
        query = self.query(query_input)

        query=F.softmax(query,dim=1)


        att_scores = key * query

        # att_map=att_map.unflatten(1,(8,8))
        # att_map=att_map*self.scale # B, c,8,h,w
        # att_map = att_map.unflatten(2, (4, 8))
        # att_map=F.softmax(att_map,dim=2)
        # att_map = att_map.flatten(2, 3)

        '''attention score'''
        # att_scores = value + att_map

        # att_scores=att_scores.flatten(1,2)
        # att_map = self.sig(att_map*self.scale)
        # x=self.att(x)

        # x = torch.cat([x, res], dim=1)

        # x = (att_map * value)

        output = self.d(att_scores)
        return output
class att_conv_LN3(nn.Module):
    def  __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None,shallow=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        self.activation=activation_function
        s=int(in_c2*8)

        self.s=s
        self.in_c2=in_c2

        self.key = nn.Sequential(
            nn.Conv2d(in_c1, 64, kernel_size=1),
        ).to('cuda')

        self.res = nn.Sequential(
            # LayerNorm2D(in_c1),
            nn.Conv2d(in_c1, 64, kernel_size=1),
            # activation_function,
            # nn.Conv2d(32, 48, kernel_size=1)
        ).to('cuda')


        self.LN = LayerNorm2D(in_c1).to('cuda')
        self.query = nn.Sequential(
            # nn.Conv2d(in_c2, s, kernel_size=1, groups=in_c2),
            nn.Conv2d(in_c2,64, kernel_size=1),
            # LayerNorm2D(32),
            # activation_function,
            # nn.Conv2d(32, 64, kernel_size=1),
        ).to('cuda')

        self.shift = nn.Sequential(
            # nn.Conv2d(in_c2, s, kernel_size=1, groups=in_c2),
            nn.Conv2d(in_c2,64, kernel_size=1),
            # LayerNorm2D(32),
            # activation_function,
            # nn.Conv2d(32, 64, kernel_size=1),
        ).to('cuda')

        self.d = nn.Sequential(
            # nn.Conv2d(in_c2, s, kernel_size=1, groups=in_c2),
            nn.Conv2d( 64, 32, kernel_size=1, bias=False),
            LayerNorm2D( 32),
            # nn.InstanceNorm2d(32),
            activation_function,
            nn.Conv2d(32, 16, kernel_size=1,bias=False),
            LayerNorm2D( 16),
            # nn.InstanceNorm2d(16),
            activation_function,
            nn.Conv2d(16, out_c, kernel_size=1),
        ).to('cuda')


    def forward(self, key_value_input, query_input):
        key_value_input = self.LN(key_value_input)
        # key_value_input = F.normalize(key_value_input, p=2, dim=1)


        '''key value from input1'''
        key = self.key(key_value_input)
        shift = self.shift(query_input)
        scale = self.query(query_input)

        scale=scale.unflatten(1,(2,-1))
        scale=F.softmax(scale,dim=1)
        scale = scale.flatten(1, 2)

        # scale=1+0.5*torch.tanh(scale)
        att_scores = key * scale

        # att_map=att_map*self.scale # B, c,8,h,w
        # att_map = att_map.unflatten(2, (4, 8))
        # att_map=F.softmax(att_map,dim=2)
        # att_map = att_map.flatten(2, 3)

        '''attention score'''
        # att_scores = value + att_map

        # att_scores=att_scores.flatten(1,2)
        # att_map = self.sig(att_map*self.scale)
        # x=self.att(x)

        # x = torch.cat([x, res], dim=1)

        # x = (att_map * value)

        output = self.d(att_scores)
        return output

class att_conv_norm_free(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0., drop_out_ratio=0.0,
                 activation=None, shallow=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        self.activation = activation_function
        s = int(in_c2 * 8)

        self.scale = nn.Parameter(torch.tensor(torch.ones(1, in_c2, 1, 1, 1), dtype=torch.float32, device='cuda'),
                                  requires_grad=True)

        # self.neck = nn.Sequential(
        #     # LayerNorm2D(in_c1),
        #     nn.Conv2d(in_c1, 32, kernel_size=1),
        #     activation_function,
        #     # nn.Conv2d(32, 48, kernel_size=1)
        # ).to('cuda')

        self.s = s
        self.in_c2 = in_c2

        self.key = nn.Sequential(
            # LayerNorm2D(in_c1),
            nn.Conv2d(in_c1, in_c1//2, kernel_size=1),
            # activation_function,
            # nn.Conv2d(32, 48, kernel_size=1)
        ).to('cuda')

        self.value = nn.Sequential(
            # LayerNorm2D(in_c1),
            nn.Conv2d(in_c1, in_c1//2, kernel_size=1),
            # activation_function,
            # nn.Conv2d(32, 48, kernel_size=1)
        ).to('cuda')

        # self.res = nn.Sequential(
        #     # LayerNorm2D(in_c1),
        #     nn.Conv2d(in_c1, 16, kernel_size=1),
        #     activation_function,
        # ).to('cuda')

        # self.att = nn.Sequential(
        #     # LayerNorm2D(in_c1),
        #     nn.Conv2d(48, 32, kernel_size=1),
        #     activation_function,
        # ).to('cuda')
        # self.LN = LayerNorm2D(in_c1).to('cuda')

        # self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        # self.value = nn.Sequential(
        #     nn.Conv2d(in_c1, 32, kernel_size=1),
        # ).to('cuda')

        self.LN = LayerNorm2D(in_c1).to('cuda')
        self.query = nn.Sequential(
            # nn.Conv2d(in_c2, s, kernel_size=1, groups=in_c2),
            nn.Conv2d(in_c2, in_c2 * (in_c1//2), kernel_size=1, groups=in_c2),
            # # LayerNorm2D(32),
            # activation_function,
            # nn.Conv2d(in_c2*4, 64, kernel_size=1),
        ).to('cuda')

        self.d = nn.Sequential(
            # nn.Conv2d(in_c2, s, kernel_size=1, groups=in_c2),
            nn.Conv2d(in_c2 * (in_c1//2), in_c2 * 12, kernel_size=1, groups=in_c2),
            nn.Dropout2d(0.),
            activation_function,
            nn.Conv2d(in_c2 * 12, 16, kernel_size=1),
            nn.Dropout2d(drop_out_ratio),
            activation_function,
            nn.Conv2d(16, out_c, kernel_size=1),
        ).to('cuda')

        # self.sig = nn.Sigmoid()

        # self.d =nn.Sequential(
        #     nn.Conv2d(64, 16, kernel_size=1),
        #     activation_function,
        #     nn.Conv2d(16, out_c, kernel_size=1),
        # ).to('cuda') if shallow  else nn.Sequential(
        #     nn.Conv2d(64, 32, kernel_size=1),
        #     activation_function,
        #     nn.Conv2d(32, 16, kernel_size=1),
        #     nn.Dropout2d(drop_out_ratio),
        #     activation_function,
        #     nn.Conv2d(16, out_c, kernel_size=1),
        # ).to('cuda')

    def forward(self, key_value_input, query_input):
        # key_value_input = self.LN(key_value_input)
        # key_value_input=self.activation(key_value_input)

        # normalzied_query_input=self.Q_LN(query_input)
        '''residual'''
        # inputs = torch.cat([key_value_input, query_input], dim=1)

        # key_value_input = self.neck(key_value_input)

        '''key value from input1'''
        key = self.key(key_value_input)
        value = self.value(key_value_input)
        query = self.query(query_input)

        # key=key.unflatten(1,(self.in_c2,8))
        # value=value.unflatten(1,(self.in_c2,8))
        key = key[:, None, ...]
        value = value[:, None, ...]

        query = query.unflatten(1, (self.in_c2, 32))

        att_map = key * query

        # att_map=att_map.unflatten(1,(8,8))
        att_map = att_map * self.scale  # B, c,8,h,w
        att_map = att_map.unflatten(2, (2, 16))
        att_map = F.softmax(att_map, dim=2)
        att_map = att_map.flatten(2, 3)

        '''attention score'''
        att_scores = value * att_map

        att_scores = att_scores.flatten(1, 2)
        # att_map = self.sig(att_map*self.scale)
        # x=self.att(x)

        # x = torch.cat([x, res], dim=1)

        # x = (att_map * value)

        output = self.d(att_scores)
        return output


class att_conv_LN_normalize(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None,shallow=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        self.activation=activation_function
        self.scale = nn.Parameter(torch.tensor(torch.ones(1,1,16,1,1), dtype=torch.float32, device='cuda'), requires_grad=True)

        self.key = nn.Sequential(
            # LayerNorm2D(in_c1),
            nn.Conv2d(in_c1, 32, kernel_size=1),
            # activation_function,
            # nn.Conv2d(32, 48, kernel_size=1)
        ).to('cuda')

        self.value = nn.Sequential(
            # LayerNorm2D(in_c1),
            nn.Conv2d(in_c1, 32, kernel_size=1),
            # activation_function,
            # nn.Conv2d(32, 48, kernel_size=1)
        ).to('cuda')


        self.LN = LayerNorm2D(in_c1).to('cuda')
        self.query = nn.Sequential(
            nn.Conv2d(in_c2, 32, kernel_size=1),

        ).to('cuda')



        # self.sig = nn.Sigmoid()


        self.d =nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1,bias=False),
            LayerNorm2D(16),
            activation_function,
            nn.Conv2d(16, out_c, kernel_size=1),
        ).to('cuda') if shallow  else nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1,bias=False),
            LayerNorm2D(16),
            activation_function,
            nn.Conv2d(16, 8, kernel_size=1,bias=False),
            LayerNorm2D(8),
            nn.Dropout2d(drop_out_ratio),
            activation_function,
            nn.Conv2d(8, out_c, kernel_size=1),
        ).to('cuda')

    def forward(self, key_value_input, query_input):
        key_value_input = self.LN(key_value_input)
        # key_value_input=self.activation(key_value_input)

        # normalzied_query_input=self.Q_LN(query_input)
        '''residual'''
        # inputs = torch.cat([key_value_input, query_input], dim=1)

        # res = self.res(key_value_input)

        '''key value from input1'''
        key = self.key(key_value_input)
        value = self.value(key_value_input)

        query = self.query(query_input)

        att_map = key * query

        att_map=att_map.unflatten(1,(2,16))
        att_map=att_map*self.scale
        att_map=F.softmax(att_map,dim=1)
        att_map=att_map.flatten(1,2)


        '''attention score'''
        att_scores = value * att_map
        # att_map = self.sig(att_map*self.scale)
        # x=self.att(x)

        # x = torch.cat([x, res], dim=1)

        # x = (att_map * value)

        output = self.d(att_scores)
        return output

class att_conv_LN_normalize_res(nn.Module):
    def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0.,  drop_out_ratio=0.0,
                 activation=None,shallow=False):
        super().__init__()
        if activation is None:
            activation_function = nn.LeakyReLU(
                negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU()
        else:
            activation_function = activation

        self.activation=activation_function
        # self.scale = nn.Parameter(torch.tensor(torch.ones(1,1,16,1,1), dtype=torch.float32, device='cuda'), requires_grad=True)

        self.key = nn.Sequential(
            # LayerNorm2D(in_c1),
            nn.Conv2d(in_c1, 32, kernel_size=1),
            # activation_function,
            # nn.Conv2d(32, 48, kernel_size=1)
        ).to('cuda')

        self.value = nn.Sequential(
            # LayerNorm2D(in_c1),
            nn.Conv2d(in_c1, 32, kernel_size=1),
            # activation_function,
            # nn.Conv2d(32, 48, kernel_size=1)
        ).to('cuda')

        self.res = nn.Sequential(
            # LayerNorm2D(in_c1),
            nn.Conv2d(in_c1, 32, kernel_size=1,bias=False),
            LayerNorm2D(32),
            activation_function,
        ).to('cuda')

        self.att = nn.Sequential(
            # LayerNorm2D(in_c1),
            # nn.Conv2d(32, 32, kernel_size=1,bias=False),
            LayerNorm2D(32),
            activation_function,
        ).to('cuda')
        # self.LN = LayerNorm2D(in_c1).to('cuda')

        # self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'), requires_grad=True)
        # self.value = nn.Sequential(
        #     nn.Conv2d(in_c1, 32, kernel_size=1),
        # ).to('cuda')

        self.LN = LayerNorm2D(in_c1).to('cuda')
        self.query = nn.Sequential(
            nn.Conv2d(in_c2, 32, kernel_size=1),
            # LayerNorm2D(16),
            # activation_function,
            # nn.Conv2d(16, 32, kernel_size=1),
        ).to('cuda')



        self.d =nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            LayerNorm2D(32),
            activation_function,
            nn.Conv2d(16, out_c, kernel_size=1),
        ).to('cuda') if shallow  else nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1,bias=False),
            LayerNorm2D(32),
            activation_function,
            nn.Conv2d(32, 16, kernel_size=1,bias=False),
            LayerNorm2D(16),
            nn.Dropout2d(drop_out_ratio),
            activation_function,
            nn.Conv2d(16, out_c, kernel_size=1),
        ).to('cuda')

        self.sig=nn.Sigmoid()

    def forward(self, key_value_input, query_input):
        key_value_input = self.LN(key_value_input)
        # key_value_input=self.activation(key_value_input)

        # normalzied_query_input=self.Q_LN(query_input)
        '''residual'''
        # inputs = torch.cat([key_value_input, query_input], dim=1)

        res = self.res(key_value_input)

        '''key value from input1'''
        key = self.key(key_value_input)
        value = self.value(key_value_input)

        query = self.query(query_input)

        att_map = key * query

        # att_map=att_map.unflatten(1,(2,16))
        # att_map=att_map*self.scale
        att_map=self.sig(att_map)
        # att_map=att_map.flatten(1,2)


        '''attention score'''
        att_scores = value * att_map
        # att_map = self.sig(att_map*self.scale)
        x=self.att(att_scores)

        x = torch.cat([x, res], dim=1)

        # x = (att_map * value)

        output = self.d(x)
        return output


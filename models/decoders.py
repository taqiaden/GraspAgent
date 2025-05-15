import torch
import torch.nn.functional as F
from torch import nn

from lib.custom_activations import GrowingCosineUnit
from models.resunet import batch_norm_relu
from visualiztion import view_features

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

class att_res_mlp_LN(nn.Module):
        def __init__(self, in_c1, in_c2, out_c, relu_negative_slope=0., shallow_decoder=False,drop_out_ratio=0.0):
            super().__init__()
            self.key = nn.Sequential(
                nn.Linear(in_c1, 32),
            ).to('cuda')

            self.value = nn.Sequential(
                nn.Linear(in_c1, 32),
            ).to('cuda')

            self.query_IN=nn.InstanceNorm1d(in_c2)

            self.query = nn.Sequential(
                # nn.LayerNorm([in_c2]),
                nn.Linear(in_c2, 32)
            ).to('cuda')

            self.res = nn.Sequential(
                nn.Linear(in_c1 + in_c2, 32, bias=False),
                nn.LayerNorm([32]),
                nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU(),
            ).to('cuda')

            self.att = nn.Sequential(
                nn.LayerNorm([32]),
                nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU(),
            ).to('cuda')

            self.d = nn.Sequential(
                nn.Linear(64, 16, bias=False),
                nn.LayerNorm([16]),
                nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU(),
                nn.Dropout(drop_out_ratio),
                nn.Linear(16, out_c),
            ).to('cuda') if shallow_decoder else nn.Sequential(
                nn.Linear(64, 32, bias=False),
                nn.LayerNorm([32]),
                nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU(),
                nn.Linear(32, 16, bias=False),
                nn.LayerNorm([16]),
                nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope > 0. else nn.ReLU(),
                nn.Dropout(drop_out_ratio),
                nn.Linear(16, out_c),
            ).to('cuda')

        def forward(self, key_value_input, query_input):
            # normalized_query_input=self.query_IN(query_input.t()).t()
            '''residual'''
            inputs = torch.cat([key_value_input, query_input], dim=-1)
            res = self.res(inputs)

            '''key value from input1'''
            key = self.key(key_value_input)
            value = self.value(key_value_input)
            '''Query from input2'''
            query = self.query(query_input)
            '''attention score'''
            att_map = key * query
            att_map = F.softmax(att_map, dim=-1)
            x = (att_map * value)
            x = self.att(x)
            x = torch.cat([x, res], dim=-1)

            output = self.d(x)
            return output


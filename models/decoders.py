import torch
from torch import nn
from models.resunet import batch_norm_relu
import torch.nn.functional as F

class res_block(nn.Module):
    def __init__(self,in_c,medium_c,out_c,Batch_norm=True,Instance_norm=False):
        super().__init__()
        self.c1 = nn.Conv2d(in_c, medium_c, kernel_size=1)
        self.b1 = batch_norm_relu(medium_c, Batch_norm=Batch_norm, Instance_norm=Instance_norm)
        self.res1 = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.c2 = nn.Conv2d(medium_c, out_c, kernel_size=1)
    def forward(self, input):
        r=self.res1(input)
        x=self.c1(input)
        x=self.b1(x)
        x=self.c2(x)
        output=x+r
        return output

class att_res_decoder_A(nn.Module):
    def __init__(self,in_c1,in_c2,out_c,Batch_norm=True,Instance_norm=False):
        super().__init__()
        assert in_c1 >32 and out_c<32 and in_c2<16

        self.key = nn.Conv2d(in_c1, 32, kernel_size=1)
        self.value = nn.Conv2d(in_c1, 32, kernel_size=1)

        self.query = res_block(in_c2,16,32,Batch_norm=Batch_norm,Instance_norm=Instance_norm)

        self.res1=res_block(in_c1+in_c2,64,32,Batch_norm=Batch_norm,Instance_norm=Instance_norm)

        self.res2=res_block(64,32,16,Batch_norm=Batch_norm,Instance_norm=Instance_norm)

        self.b1 = batch_norm_relu(16, Batch_norm=Batch_norm, Instance_norm=Instance_norm)
        self.c1 = nn.Conv2d(16, out_c, kernel_size=1)
    def forward(self, input1,input2):
        '''residual'''
        inputs=torch.cat([input1,input2],dim=1)
        res=self.res1(inputs)

        '''key value from input1'''
        key=self.key(input1)
        value=self.value(input1)

        '''Query from input2'''
        query=self.query(input2)

        '''attention score'''
        att_map=key*query
        att_map=F.softmax(att_map,dim=1)
        att_score=att_map*value

        '''output'''
        embedding=torch.cat([att_score,res],dim=1)
        x=self.res2(embedding)
        x=self.b1(x)
        output=self.c1(x)
        return output

class decoder2(nn.Module):
    def __init__(self,in_c,out_c,Batch_norm=True,Instance_norm=False):
        super().__init__()
        assert in_c >32 and out_c<32
        self.c1 = nn.Conv2d(in_c, 32, kernel_size=1)
        self.b1 = batch_norm_relu(32,Batch_norm=Batch_norm,Instance_norm=Instance_norm)
        self.c2 = nn.Conv2d(32, out_c, kernel_size=1)
    def forward(self, input):
        x=self.c1(input)
        x=self.b1(x)
        output=self.c2(x)
        return output
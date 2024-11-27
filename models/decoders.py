import torch
from torch import nn
from models.resunet import batch_norm_relu
import torch.nn.functional as F

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
    def __init__(self,in_c,medium_c,out_c,drop_out_ratio=0.0,activation=nn.ReLU()):
        super().__init__()
        self.ln1=nn.LayerNorm([in_c])

        self.c1 = nn.Linear(in_c, medium_c)

        self.ln2=nn.LayerNorm([medium_c])
        self.res1 = nn.Linear(in_c, out_c)
        self.c2 = nn.Linear(medium_c, out_c)
        self.drop_out=nn.Dropout(drop_out_ratio)

        self.activation = activation
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
# nn.Linear(128, 64, bias=False),
#             nn.LayerNorm([64]),

class att_res_mlp_LN(nn.Module):
    def __init__(self,in_c1,in_c2,out_c,drop_out_ratio=0.0):
        super().__init__()
        assert in_c1 >32 and out_c<32 and in_c2<16

        self.key = nn.Sequential(
            nn.Linear(in_c1, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(),
            nn.Linear(32, 16),
        ).to('cuda')

        self.value = nn.Sequential(
            nn.Linear(in_c1, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(),
            nn.Linear(32, 16),
        ).to('cuda')

        self.query =  nn.Sequential(
            nn.Linear(in_c2, 12, bias=False),
            nn.LayerNorm([12]),
            nn.ReLU(),
            nn.Linear(12, 16),
        ).to('cuda')

        self.res=nn.Sequential(
            nn.Linear(in_c1+in_c2, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(),
            nn.Linear(32, 16),
        ).to('cuda')

        self.d=nn.Sequential(
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.ReLU(),
            nn.Dropout(drop_out_ratio),
            nn.Linear(16, out_c),
        ).to('cuda')



    def forward(self, key_value_input,query_input):
        '''residual'''
        inputs=torch.cat([key_value_input,query_input],dim=1)
        res=self.res(inputs)

        '''key value from input1'''
        key=self.key(key_value_input)
        value=self.value(key_value_input)

        '''Query from input2'''
        query=self.query(query_input)

        '''attention score'''
        att_map=key*query
        att_map=F.softmax(att_map,dim=1)
        att_score=att_map*value

        '''output'''
        embedding=torch.cat([att_score,res],dim=1)
        output=self.d(embedding)
        return output


class att_res_decoder_A(nn.Module):
    def __init__(self,in_c1,in_c2,out_c,Batch_norm=True,Instance_norm=False):
        super().__init__()
        assert in_c1 >32 and out_c<32 and in_c2<16

        self.key = nn.Conv2d(in_c1, 32, kernel_size=1)
        self.value = nn.Conv2d(in_c1, 32, kernel_size=1)

        self.query = res_block(in_c2,16,32,Batch_norm=Batch_norm,Instance_norm=Instance_norm)
        self.att = nn.Conv2d(32, 32, kernel_size=1)

        self.res1=res_block(in_c1+in_c2,64,32,Batch_norm=Batch_norm,Instance_norm=Instance_norm)

        self.res2=res_block(64,32,16,Batch_norm=Batch_norm,Instance_norm=Instance_norm)

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
        att_score=self.att(att_score)

        '''output'''
        embedding=torch.cat([att_score,res],dim=1)
        x=self.res2(embedding)
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
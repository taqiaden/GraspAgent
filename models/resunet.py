import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from lib.image_utils import view_image
from lib.models_utils import number_of_parameters


class batch_norm_relu(nn.Module):
    def __init__(self, in_c,Batch_norm=True,Instance_norm=False,relu_negative_slope=0.,activation=None,IN_affine=False):
        super().__init__()
        self.batch_norm=Batch_norm
        self.instance_norm=Instance_norm
        self.Bn = nn.BatchNorm2d(in_c)
        self.In = nn.InstanceNorm2d(in_c,affine=IN_affine)
        if activation is None:
            self.relu = nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope>0. else nn.ReLU()
        else:
            self.relu=activation

    def forward(self, inputs):
        x=inputs
        if self.batch_norm is not None and self.batch_norm:
            x = self.Bn(x)
        if self.instance_norm is not None and self.instance_norm:
            x = self.In(x)
        x = self.relu(x)
        return x

class residual_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1,Batch_norm=True,Instance_norm=False,relu_negative_slope=0.,activation=None,IN_affine=False,scale=1.0):
        super().__init__()
        """ Convolutional layer """
        self.b1 = batch_norm_relu(in_c,Batch_norm,Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation,IN_affine=IN_affine)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.b2 = batch_norm_relu(out_c,Batch_norm,Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation,IN_affine=IN_affine)
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)

        """ Shortcut Connection (Identity Mapping) """
        self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)

        self.scale=1. if scale is None else nn.Parameter(torch.tensor(scale, dtype=torch.float32, device='cuda'), requires_grad=True)


    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.b2(x)
        x = self.c2(x)
        s = self.s(inputs)

        skip = x*self.scale + s
        return skip
# wget https://repo.continuum.io/archive/Anaconda3-5.3.1-Linux-x86_64.sh
# bash archive/Anaconda3-5.3.1-Linux-x86_64.sh -b -p ~/anaconda
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c,Batch_norm=True,Instance_norm=False,relu_negative_slope=0.,activation=None,IN_affine=False,scale=1.0):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r = residual_block(in_c+out_c, out_c,Batch_norm=Batch_norm,Instance_norm=Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation,IN_affine=IN_affine,scale=scale)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r(x)
        return x

def add_spectral_norm_selective(model, layer_types=(nn.Conv2d, nn.Linear)):
    for name, layer in model.named_children():
        if isinstance(layer, layer_types):
            setattr(model, name, spectral_norm(layer, name='weight'))
        else:
            add_spectral_norm_selective(layer, layer_types)
    return model

class res_unet(nn.Module):
    def __init__(self,in_c,Batch_norm=True,Instance_norm=False,relu_negative_slope=0.0,activation=None,IN_affine=False,scale=None):
        super().__init__()

        """ Encoder 1 """
        self.c11 = nn.Conv2d(in_c, 64, kernel_size=3, padding=1)
        self.br1 = batch_norm_relu(64,Batch_norm,Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation,IN_affine=IN_affine)
        self.c12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(in_c, 64, kernel_size=1, padding=0)

        """ Encoder 2 and 3 """
        self.r2 = residual_block(64, 128, stride=2,Batch_norm=Batch_norm,Instance_norm=Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation,IN_affine=IN_affine,scale=scale)
        self.r3 = residual_block(128, 256, stride=2,Batch_norm=Batch_norm,Instance_norm=Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation,IN_affine=IN_affine,scale=scale)

        """ Bridge """
        self.r4 = residual_block(256, 512, stride=2,Batch_norm=Batch_norm,Instance_norm=Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation,IN_affine=IN_affine,scale=scale)

        """ Decoder """
        self.d1 = decoder_block(512, 256,Batch_norm,Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation,IN_affine=IN_affine,scale=scale)
        self.d2 = decoder_block(256, 128,Batch_norm,Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation,IN_affine=IN_affine,scale=scale)
        self.d3 = decoder_block(128, 64,Batch_norm,Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation,IN_affine=IN_affine,scale=scale)

        """ Output """
        # self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        # self.sigmoid = nn.Sigmoid()

    def SN_on_encoder(self):
        add_spectral_norm_selective(self.c11)
        add_spectral_norm_selective(self.c12)
        add_spectral_norm_selective(self.c13)
        add_spectral_norm_selective(self.r2)
        add_spectral_norm_selective(self.r3)
        add_spectral_norm_selective(self.r4)


    def forward(self, inputs):
        """ Encoder 1 """
        x = self.c11(inputs)
        x = self.br1(x)

        x = self.c12(x)

        s = self.c13(inputs)
        skip1 = x + s

        # print(x.shape)

        """ Encoder 2 and 3 """
        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)

        """ Bridge """
        b = self.r4(skip3)

        # print(skip2.shape)
        # print(skip3.shape)
        # print(b.shape)
        # exit()

        """ Decoder """
        d1 = self.d1(b, skip3)

        d2 = self.d2(d1, skip2)
        """ output """
        output = self.d3(d2, skip1)
        return output

class res_unet_encoder(nn.Module):
    def __init__(self, in_c, Batch_norm=True, Instance_norm=False, relu_negative_slope=0.0, activation=None,
                 IN_affine=False, output_size=64,scale=1.0):
        super().__init__()
        """ Encoder 1 """
        self.c11 = nn.Conv2d(in_c, 64, kernel_size=3, padding=1)
        self.br1 = batch_norm_relu(64,Batch_norm,Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation,IN_affine=IN_affine)
        self.c12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(in_c, 64, kernel_size=1, padding=0)

        """ Encoder 2 and 3 """
        self.r2 = residual_block(64, 128, stride=2,Batch_norm=Batch_norm,Instance_norm=Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation,IN_affine=IN_affine,scale=scale)
        self.r3 = residual_block(128, 256, stride=2,Batch_norm=Batch_norm,Instance_norm=Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation,IN_affine=IN_affine,scale=scale)

        """ Bridge """
        self.r4 = residual_block(256, 512, stride=2,Batch_norm=Batch_norm,Instance_norm=Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation,IN_affine=IN_affine,scale=scale)

        self.pool=nn.AdaptiveAvgPool2d((1,1))

        self.d=nn.Conv2d(512,output_size,kernel_size=1)

    def forward(self, inputs):
        """ Encoder 1 """
        x = self.c11(inputs)
        x = self.br1(x)

        x = self.c12(x)

        s = self.c13(inputs)
        skip1 = x + s

        # print(x.shape)

        """ Encoder 2 and 3 """
        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)

        """ Bridge """
        x = self.r4(skip3)

        x=self.pool(x)
        output=self.d(x)


        return output

class ResNet(nn.Module):
    def __init__(self,in_c,Batch_norm=True,Instance_norm=False,relu_negative_slope=0.0,activation=None,IN_affine=False):
        super().__init__()

        """ Encoder 1 """
        self.c11 = nn.Conv2d(in_c, 64, kernel_size=3, padding=1)
        self.br1 = batch_norm_relu(64,Batch_norm,Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation,IN_affine=IN_affine)
        self.c12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(in_c, 64, kernel_size=1, padding=0)

        """ Encoder 2 and 3 """
        self.r2 = residual_block(64, 128, stride=2,Batch_norm=Batch_norm,Instance_norm=Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation,IN_affine=IN_affine)
        self.r3 = residual_block(128, 256, stride=2,Batch_norm=Batch_norm,Instance_norm=Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation,IN_affine=IN_affine)

        """ Bridge """
        self.r4 = residual_block(256, 512, stride=2,Batch_norm=Batch_norm,Instance_norm=Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation,IN_affine=IN_affine)



    def forward(self, inputs):
        """ Encoder 1 """
        x = self.c11(inputs)
        x = self.br1(x)

        x = self.c12(x)

        s = self.c13(inputs)
        skip1 = x + s

        """ Encoder 2 and 3 """
        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)

        """ Bridge """
        output = self.r4(skip3)


        return output

if __name__ == "__main__":
    inputs = torch.randn((2, 4, 480, 712)).to('cuda')
    model = res_unet(4).to('cuda')
    # with torch.no_grad():
    y = model(inputs)
    p=number_of_parameters(model)
    print(p)
    print(y.shape)
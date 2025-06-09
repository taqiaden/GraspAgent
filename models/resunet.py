import torch
import torch.nn as nn

from lib.image_utils import view_image
from lib.models_utils import number_of_parameters


class batch_norm_relu(nn.Module):
    def __init__(self, in_c,Batch_norm=True,Instance_norm=False,relu_negative_slope=0.,activation=None):
        super().__init__()
        self.batch_norm=Batch_norm
        self.instance_norm=Instance_norm
        self.Bn = nn.BatchNorm2d(in_c)
        self.In = nn.InstanceNorm2d(in_c)
        if activation is None:
            self.relu = nn.LeakyReLU(negative_slope=relu_negative_slope) if relu_negative_slope>0. else nn.ReLU()
        else:
            self.relu=activation

    def forward(self, inputs):
        x=inputs
        if self.batch_norm:
            x = self.Bn(x)
        if self.instance_norm:
            x = self.In(x)
        x = self.relu(x)
        return x

class residual_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1,Batch_norm=True,Instance_norm=False,relu_negative_slope=0.,activation=None):
        super().__init__()
        """ Convolutional layer """
        self.b1 = batch_norm_relu(in_c,Batch_norm,Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.b2 = batch_norm_relu(out_c,Batch_norm,Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation)
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)

        """ Shortcut Connection (Identity Mapping) """
        self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.b2(x)
        x = self.c2(x)
        s = self.s(inputs)

        skip = x + s
        return skip
# wget https://repo.continuum.io/archive/Anaconda3-5.3.1-Linux-x86_64.sh
# bash archive/Anaconda3-5.3.1-Linux-x86_64.sh -b -p ~/anaconda
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c,Batch_norm=True,Instance_norm=False,relu_negative_slope=0.,activation=None):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r = residual_block(in_c+out_c, out_c,Batch_norm=Batch_norm,Instance_norm=Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r(x)
        return x

class res_unet(nn.Module):
    def __init__(self,in_c,Batch_norm=True,Instance_norm=False,relu_negative_slope=0.0,activation=None):
        super().__init__()

        """ Encoder 1 """
        self.c11 = nn.Conv2d(in_c, 64, kernel_size=3, padding=1)
        self.br1 = batch_norm_relu(64,Batch_norm,Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation)
        self.c12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(in_c, 64, kernel_size=1, padding=0)

        """ Encoder 2 and 3 """
        self.r2 = residual_block(64, 128, stride=2,Batch_norm=Batch_norm,Instance_norm=Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation)
        self.r3 = residual_block(128, 256, stride=2,Batch_norm=Batch_norm,Instance_norm=Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation)

        """ Bridge """
        self.r4 = residual_block(256, 512, stride=2,Batch_norm=Batch_norm,Instance_norm=Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation)

        """ Decoder """
        self.d1 = decoder_block(512, 256,Batch_norm,Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation)
        self.d2 = decoder_block(256, 128,Batch_norm,Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation)
        self.d3 = decoder_block(128, 64,Batch_norm,Instance_norm,relu_negative_slope=relu_negative_slope,activation=activation)

        """ Output """
        # self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        # self.sigmoid = nn.Sigmoid()

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
        b = self.r4(skip3)

        """ Decoder """
        d1 = self.d1(b, skip3)

        d2 = self.d2(d1, skip2)
        """ output """
        output = self.d3(d2, skip1)
        return output

if __name__ == "__main__":
    inputs = torch.randn((2, 4, 480, 712)).to('cuda')
    model = res_unet(4).to('cuda')
    # with torch.no_grad():
    y = model(inputs)
    p=number_of_parameters(model)
    print(p)
    print(y.shape)
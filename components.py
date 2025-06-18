from pytorch_wavelets import DWTForward
import random
import math
from dca import *
from swin2 import *
from EMA import *


class depthwise_conv(nn.Module):
    def __init__(self,in_channel,out_channel,):
        super(depthwise_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,in_channel,kernel_size=3,stride=1,padding=1,groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            # ScConv(op_channel=in_channel),
            # ECA_layer(in_channel),
            # EMA(in_channel),
            nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(dropout_rate),
            # ECA_layer(out_channel),


        )

    def forward(self, x):

        return self.conv(x)



# down
class HWDownsampling(nn.Module):
    def __init__(self, in_channel, out_channel,):
        super(HWDownsampling, self).__init__()
        self.wt = DWTForward(J=1, wave='haar', mode='zero')

        self.conv_bn_relu = nn.Sequential(
                nn.Conv2d(in_channel * 4, out_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
        )


    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x


# resnet
class Residual(nn.Module):
    def __init__(self, in_size, out_size,use_1x1conv=True,stride=1,dropout_rate=None):
        super(Residual, self).__init__()

        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride,padding=1)

        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.conv3 = None

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_size, out_size, kernel_size=1, stride=stride)
        if dropout_rate:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = None

    def forward(self, X):
        out = F.relu(self.bn1(self.conv1(X)))
        out = self.bn2(self.conv2(out))
        if self.conv3 is not None:
            X = self.conv3(X)
        out += X
        out = F.relu(out)
        if self.dropout is not None:
            out = self.dropout(out)

        return out



class ResnetBlock(nn.Module):
    def __init__(self, in_size, out_size,num_residuals,dropout_rate=None):
        super(ResnetBlock, self).__init__()
        self.residual_list = nn.ModuleList()
        if dropout_rate:
            for i in range(num_residuals):
                if i == 0:
                    self.residual_list.append(
                        Residual(in_size, out_size, use_1x1conv=True, stride=1,
                                 dropout_rate=math.sqrt((i+1)/num_residuals)*dropout_rate))
                else:
                    self.residual_list.append(
                        Residual(out_size, out_size, use_1x1conv=False, stride=1,
                                 dropout_rate=math.sqrt((i+1)/num_residuals)*dropout_rate))
        else:
            for i in range(num_residuals):
                if i == 0:
                    self.residual_list.append(Residual(in_size, out_size, use_1x1conv=True, stride=1,dropout_rate=dropout_rate))
                else:
                    self.residual_list.append(Residual(out_size, out_size, use_1x1conv=False, stride=1,dropout_rate=dropout_rate))

    def forward(self, X):
        for i in range(len(self.residual_list)):
            X = self.residual_list[i](X)

        return X

#densenet
class my_conv(nn.Module):
    def __init__(self,in_channels,growth_rate,dropout_rate=None):
        super(my_conv,self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1x1 = nn.Conv2d(in_channels,4*growth_rate,kernel_size=1,stride=1,bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv3x3 = nn.Conv2d(4*growth_rate,growth_rate,kernel_size=3,stride=1,padding=1,bias=False)
        if dropout_rate:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = None
    def forward(self, x):
        x1 = self.conv1x1(F.relu(self.bn1(x)))
        x1 = self.conv3x3(F.relu(self.bn2(x1)))
        if self.dropout is not None:
            x1 = self.dropout(x1)
        x = torch.cat([x,x1],1)
        return x

class DenseBlock(nn.Module):
    def __init__(self,in_channels,increase_k,layers_num,max_dropout_rate=None):
        super(DenseBlock, self).__init__()
        self.layer_list = nn.ModuleList()
        if max_dropout_rate:
            for i in range(layers_num):
                self.layer_list.append(my_conv(in_channels+i*increase_k,increase_k,
                                               dropout_rate=math.sqrt((i+1)/layers_num)*max_dropout_rate))
        else:
            for i in range(layers_num):
                self.layer_list.append(my_conv(in_channels+i*increase_k,increase_k))

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x



def print_parameters(model):
    """递归打印模型各子模块的参数量统计"""
    total = sum(p.numel() for p in model.parameters())

    # 存储各层参数信息
    params_list = []

    def _get_params(module, name, prefix=''):
        nonlocal params_list
        full_name = f"{prefix}.{name}" if prefix else name
        num_params = sum(p.numel() for p in module.parameters())

        # 仅统计叶子模块（没有子模块的模块）
        if len(list(module.children())) == 0:
            params_list.append((full_name, num_params))

        for child_name, child_module in module.named_children():
            _get_params(child_module, child_name, full_name)

    _get_params(model, 'UNet')

    # 打印统计结果
    print(f"{'Module':<40} | {'Params':>15} | {'% Total':>8}")
    print("-" * 70)
    for name, num in sorted(params_list, key=lambda x: -x[1]):
        if num == 0: continue
        print(f"{name:<40} | {num:15,} | {100 * num / total:6.2f}%")

    print("-" * 70)
    print(f"{'Total Parameters':<40} | {total:15,} | {'100%':>8}")



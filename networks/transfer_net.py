
import torch.nn as nn
import torch
import torch.nn.functional as F



class TransformerNet(torch.nn.Module):
    def __init__(self, style_num):
        super(TransformerNet, self).__init__()

        self.style_bank = Style_bank(style_num)

        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = InstanceNorm2d(32)
        self.cm1 = condition_modulate(32)

        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = InstanceNorm2d(64)
        self.cm2 = condition_modulate(64)

        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = InstanceNorm2d(128)
        self.cm3 = condition_modulate(128)

        self.res1 = ResidualBlock(channels=128,dynamic_channels=128,groups=128)
        self.res2 = ResidualBlock(channels=128,dynamic_channels=128,groups=128)
        self.res3 = ResidualBlock(channels=128,dynamic_channels=128,groups=128)
        self.res4 = ResidualBlock(channels=128,dynamic_channels=128,groups=128)
        self.res5 = ResidualBlock(channels=128,dynamic_channels=128,groups=128)


        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = InstanceNorm2d(64)
        self.cm4 = condition_modulate(64)

        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = InstanceNorm2d(32)
        self.cm5 = condition_modulate(32)

        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.relu = torch.nn.ReLU()


    def forward(self, X, style_id):

        representation = self.style_bank(style_id)

        y = self.conv1(X)
        y = self.in1(y)
        y = self.cm1(y,representation) # conditional modulated
        y = self.relu(y)


        y = self.conv2(y)
        y = self.in2(y)
        y = self.cm2(y, representation)  # conditional modulated
        y = self.relu(y)

        y = self.conv3(y)
        y = self.in3(y)
        y = self.cm3(y, representation)  # conditional modulated
        y = self.relu(y)

        y = self.res1(y, representation)
        y = self.res2(y, representation)
        y = self.res3(y, representation)
        y = self.res4(y, representation)
        y = self.res5(y, representation)

        y = self.deconv1(y)
        y = self.in4(y)
        y = self.cm4(y, representation)  # conditional modulated
        y = self.relu(y)

        y = self.deconv2(y)
        y = self.in5(y)
        y = self.cm5(y, representation)  # conditional modulated
        y = self.relu(y)

        y = self.deconv3(y)



        return y,representation


class condition_modulate(torch.nn.Module):
    """
    Conditional Instance Normalization
    introduced in https://arxiv.org/abs/1610.07629
    created and applied based on my limited understanding, could be improved
    """

    def __init__(self, in_channels):
        super(condition_modulate, self).__init__()
        self.compress_gamma = torch.nn.Sequential(
            torch.nn.Linear(32, in_channels,bias=False),
            torch.nn.LeakyReLU(0.1, True)
        )
        self.compress_beta = torch.nn.Sequential(
            torch.nn.Linear(32, in_channels, bias=False),
            torch.nn.LeakyReLU(0.1, True)
        )

    def forward(self, x,representation):
        gamma = self.compress_gamma(representation)
        beta = self.compress_beta(representation)

        b,c = gamma.size()

        gamma = gamma.view(b,c,1,1)
        beta = beta.view(b,c,1,1)

        out = x * gamma + beta
        return out




class InstanceNorm2d(torch.nn.Module):
    """
    Conditional Instance Normalization
    introduced in https://arxiv.org/abs/1610.07629
    created and applied based on my limited understanding, could be improved
    """

    def __init__(self, in_channels):
        super(InstanceNorm2d, self).__init__()
        self.inns = torch.nn.InstanceNorm2d(in_channels, affine=False)

    def forward(self, x):
        out = self.inns(x)
        return out

class Dynamic_ConvLayer2(torch.nn.Module):
    '''
    in_channels: 输入的图像特征通道数
    out_channels: 输出的图像特征通道数
    groups: 分组数（in_channels，out_channels的公因数）
    '''
    def __init__(self, in_channels, out_channels, kernel_size, groups):
        super(Dynamic_ConvLayer2, self).__init__()
        reflection_padding = kernel_size // 2  # same dimension after padding
        self.reflection_padding = reflection_padding
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)

        self.kernel_size = kernel_size

        self.compress_key = torch.nn.Sequential(
            torch.nn.Linear(32, out_channels * kernel_size * kernel_size, bias=False),
            torch.nn.LeakyReLU(0.1, True)
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

    def forward(self, x,representation):
        out = self.reflection_pad(x)

        b, c, h, w = out.size()

        kernel = self.compress_key(representation).view(b,self.out_channels, -1, self.kernel_size, self.kernel_size)

        # 1,64,1,kh,kw -> 1,64,4,kh,kw
        features_per_group = int(self.in_channels/self.groups)
        kernel = kernel.repeat_interleave(features_per_group, dim=2)

        # 1,64,4,kh,kw
        k_batch,k_outputchannel,k_feature_pergroup,kh,kw = kernel.size()

        out = F.conv2d(out.view(1, -1, h, w), kernel.view(-1,k_feature_pergroup,kh,kw), groups=b * self.groups, padding=0)

        b,c,h,w = x.size()
        out = out.view(b, -1, h, w)

        return out



class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2  # same dimension after padding
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)  # remember this dimension

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels,dynamic_channels,groups):
        super(ResidualBlock, self).__init__()
        self.conv1 = Dynamic_ConvLayer2(channels, dynamic_channels, kernel_size=3, groups=groups)
        self.in1 = InstanceNorm2d(dynamic_channels)
        self.cm1 = condition_modulate(dynamic_channels)

        self.conv2 = ConvLayer(dynamic_channels, channels, kernel_size=1, stride=1)
        self.in2 = InstanceNorm2d(channels)
        self.cm2 = condition_modulate(channels)

        self.relu = torch.nn.ReLU()

        representation_channels = 32
        feature_channels = channels
        self.ca = CA_layer(channels_in=representation_channels, channels_out=feature_channels, reduction=4)

    def forward(self, x,representation):
        residual = x

        out = self.conv1(x,representation)
        out = self.in1(out)
        out = self.cm1(out, representation)  # conditional modulated

        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.cm2(out, representation)  # conditional modulated

        # out = out + residual
        out = out + self.ca([residual, representation])

        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(mode='nearest', scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out





class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.PReLU(),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        att = self.conv_du(x[1][:, :, None, None])

        return x[0] * att





class style_representation(nn.Module):
    def __init__(self):
        super(style_representation, self).__init__()
        params = torch.ones(32, requires_grad=True).cuda()
        self.params = nn.Parameter(params)


    def forward(self):

        z = torch.normal(mean=0., std=0.1, size=(32,),requires_grad=False).cuda() # todo:加噪修改
        y = self.params + z # todo:加噪修改
        return y



class Style_bank(nn.Module):
    def __init__(self, total_style):
        super(Style_bank, self).__init__()

        self.total_style = total_style

        self.style_para_list = nn.ModuleList()
        for i in range(total_style + 1):
            params_layer = style_representation()
            self.style_para_list.append(params_layer)



    def forward(self, style_id=None):
        new_z = []
        if style_id is not None:
            for idx, i in enumerate(style_id):
                zs = self.style_para_list[i]()
                new_z.append(zs)
            # z = torch.cat(new_z, dim=0)
            new_z = torch.stack(new_z,dim=0)
        else:
            print('where is your style_id?')
            exit(111)

        all_z = [self.style_para_list[i]() for i in range(len(self.style_para_list))]
        all_z = torch.stack(all_z,dim=0)
        return new_z

    def add_style(self,add_num):

        origin_style_num = len(self.style_para_list) # 包含ae representation
        for i in range(origin_style_num):
            self.style_para_list[i].params.requires_grad_(False)

        for i in range(add_num):
            print('add a style in bank, style id:',len(self.style_para_list))
            params_layer = style_representation()
            self.style_para_list.append(params_layer)






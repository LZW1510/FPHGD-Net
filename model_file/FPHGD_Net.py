from Data_Utils import *


class Residual_Block(nn.Module):

    def __init__(self, input_channel):
        super(Residual_Block, self).__init__()

        self.Convolution_3_1 = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1)
        self.ReLU = nn.ReLU()
        self.Convolution_3_2 = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1)

    def forward(self, x_input):
        return self.Convolution_3_2(self.ReLU(self.Convolution_3_1(x_input))) + x_input


class F_function(nn.Module):

    def __init__(self, input_channel):
        super(F_function, self).__init__()

        self.Convolution_3_1 = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1)
        self.Residual_Block = Residual_Block(input_channel)

    def forward(self, x_input):
        return self.Convolution_3_1(self.Residual_Block(x_input))


class F_T_function(nn.Module):

    def __init__(self, input_channel):
        super(F_T_function, self).__init__()

        self.Convolution_3_1 = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1)
        self.Residual_Block = Residual_Block(input_channel)

    def forward(self, x_input):
        return self.Residual_Block(self.Convolution_3_1(x_input))


class FGDM_Module(nn.Module):

    def __init__(self, input_channel):
        super(FGDM_Module, self).__init__()

        self.input_channel = input_channel

        self.F_function = F_function(self.input_channel)
        self.F_T_function = F_T_function(self.input_channel)

        self.Residual_Block = Residual_Block(input_channel)

    def forward(self, x_input, Phi, PhiT, PhiTb):
        x = self.F_function(x_input)

        x_b = Channel_2_Batch(x)

        error = PhiTb - PhiTPhifun(PhiT, Phi, x_b)
        error = Batch_2_Channel(error, self.input_channel)
        error = self.F_T_function(error)

        update_value = x_input + error

        x = self.Residual_Block(update_value)

        return x


class FPN_Module(nn.Module):

    def __init__(self, input_channel):
        super(FPN_Module, self).__init__()

        self.Convolution_3_1 = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1)
        self.Residual_Block_1 = Residual_Block(input_channel)

        self.Down_Sample_1 = nn.Conv2d(input_channel, input_channel * 2, kernel_size=3, padding=1, stride=2)
        self.Residual_Block_2 = Residual_Block(input_channel * 2)

        self.Down_Sample_2 = nn.Conv2d(input_channel * 2, input_channel * 4, kernel_size=3, padding=1, stride=2)
        self.Residual_Block_3 = Residual_Block(input_channel * 4)

        self.Up_Sample_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.Convolution_3_2 = nn.Conv2d(input_channel * 6, input_channel * 2, kernel_size=3, padding=1)
        self.Residual_Block_4 = Residual_Block(input_channel * 2)

        self.Up_Sample_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.Convolution_3_3 = nn.Conv2d(input_channel * 3, input_channel, kernel_size=3, padding=1)
        self.Residual_Block_5 = Residual_Block(input_channel)

    def forward(self, x_input):

        x_1 = self.Convolution_3_1(x_input)
        x_1 = self.Residual_Block_1(x_1)

        x_2 = self.Down_Sample_1(x_1)
        x_2 = self.Residual_Block_2(x_2)

        x_3 = self.Down_Sample_2(x_2)
        x_3 = self.Residual_Block_3(x_3)

        x_3 = self.Up_Sample_1(x_3)
        x_4 = torch.cat((x_2, x_3), dim=1)
        x_4 = self.Convolution_3_2(x_4)
        x_4 = self.Residual_Block_4(x_4)

        x_4 = self.Up_Sample_2(x_4)
        x_5 = torch.cat((x_1, x_4), dim=1)
        x_5 = self.Convolution_3_3(x_5)
        x_m = self.Residual_Block_5(x_5)


        return x_m


class IF_Module(nn.Module):

    def __init__(self, input_channel):
        super(IF_Module, self).__init__()

        self.Convolution_3_1 = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1)
        self.Convolution_3_2 = nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1)
        self.Residual_Block = Residual_Block(input_channel)
        self.Relu = nn.ReLU()

    def forward(self, x_input, x_e):
        x_e = self.Convolution_3_1(x_e)
        x_e = self.Relu(x_e)
        x_e = self.Convolution_3_2(x_e)
        x = x_input + x_e
        x = self.Residual_Block(x)

        return x


class CS_Reconstruction(nn.Module):

    def __init__(self, input_channel, cs_ratio, layer_num):

        super(CS_Reconstruction, self).__init__()

        self.input_channel = input_channel
        self.measurement = int(1024 * cs_ratio)
        self.layer_num = layer_num

        self.Phi = nn.Parameter(init.kaiming_normal_(torch.Tensor(self.measurement, 1024)))

        self.Convolution_3_1 = nn.Conv2d(1, self.input_channel, kernel_size=3, padding=1)
        self.Residual_Block_1 = Residual_Block(self.input_channel)

        self.Residual_Block_D = Residual_Block(self.input_channel)
        self.Feature_Pyramid_Network = FPN_Module(self.input_channel)

        layer_1 = []
        for i in range(self.layer_num):
            layer_1.append(FGDM_Module(self.input_channel))
        self.Iterative_Reconstruction = nn.ModuleList(layer_1)

        layer_2 = []
        for i in range(self.layer_num):
            layer_2.append(IF_Module(input_channel))
        self.IF_Reconstruction = nn.ModuleList(layer_2)

        self.Convolution_3_2 = nn.Conv2d(self.input_channel, 1, kernel_size=3, padding=1)
        self.Residual_Block_2 = Residual_Block(self.input_channel)

        self.Dropout = nn.Dropout(p=0)

    def forward(self, batch_x):

        Phi = self.Phi.contiguous().view(-1, 1, 32, 32)
        Phix = F.conv2d(batch_x, Phi, stride=32, padding=0, bias=None)
        PhiT = self.Phi.t().contiguous().view(1024, -1, 1, 1)
        PhiTb = F.conv2d(Phix, PhiT, stride=1, padding=0, bias=None)
        PhiTb = nn.PixelShuffle(32)(PhiTb)

        x = self.Convolution_3_1(PhiTb)
        x = self.Dropout(x)
        x = self.Residual_Block_1(x)

        x = self.Residual_Block_D(x)

        PhiTb_C2B = Channel_2_Batch(PhiTb.repeat(1, self.input_channel, 1, 1))
        for i in range(self.layer_num):
            x_e = self.Feature_Pyramid_Network(x)
            x = self.IF_Reconstruction[i](x, x_e)
            x = self.Iterative_Reconstruction[i](x, Phi, PhiT, PhiTb_C2B)

        x = self.Residual_Block_2(x)
        x = self.Convolution_3_2(x)

        return x, self.Phi


if __name__ == '__main__':
    model = CS_Reconstruction(16, 0.5, 12)
    total = sum([p.nelement() for p in model.parameters()])
    print(total)
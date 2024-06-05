from Data_Utils import *


def save_image(x, phase, which):
    save_path = f'./check/{phase}/{which}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    l = x.size(1)
    x = x.squeeze(0)
    x = x.cpu().data.numpy()
    for i in range(l):
        p = x[i, :, :]
        p = p * 255
        p = np.clip(p, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_path, f'{which}_{i}.png'), p)


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

        self.Residual_Block_1 = Residual_Block(input_channel)
        self.Residual_Block_2 = Residual_Block(input_channel)

    def forward(self, x_input, Phi, PhiT, PhiTb, i):

        x = self.F_function(x_input)

        x_b = Channel_2_Batch(x)

        error = PhiTb - PhiTPhifun(PhiT, Phi, x_b)
        error = Batch_2_Channel(error, self.input_channel)
        error = self.F_T_function(error)

        update_value = x_input + error

        x = self.Residual_Block_1(update_value)
        x = self.Residual_Block_2(x)

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


        self.Residual_Block_D = nn.Sequential(
            Residual_Block(self.input_channel),
            Residual_Block(self.input_channel)
        )

        layer_1 = []
        for i in range(self.layer_num):
            layer_1.append(FGDM_Module(self.input_channel))
        self.Iterative_Reconstruction = nn.ModuleList(layer_1)

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
        x = self.Residual_Block_1(x)

        x = self.Residual_Block_D(x)

        PhiTb_C2B = Channel_2_Batch(PhiTb.repeat(1, self.input_channel, 1, 1))

        for i in range(self.layer_num):
            x = self.Iterative_Reconstruction[i](x, Phi, PhiT, PhiTb_C2B, i)

        x = self.Residual_Block_2(x)
        x = self.Convolution_3_2(x)

        return x, Phix


if __name__ == '__main__':
    model = CS_Reconstruction(16, 0.5, 16)
    total = sum([p.nelement() for p in model.parameters()])
    print(total)



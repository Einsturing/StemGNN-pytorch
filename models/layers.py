import torch
import torch.nn as nn
import torch.nn.functional as F


def graph_fft(x, v, flag=True):
    _, T, n, c_in = x.shape[:]
    if flag:
        U = v.permute(1, 0)
    else:
        U = v
    x_tmp = torch.reshape(x.permute(2, 0, 1, 3), (n, -1))
    x = torch.matmul(U, x_tmp)
    x = torch.reshape(x, (n, -1, T, c_in)).permute(1, 2, 0, 3)
    return x


class stemGNN_block(nn.Module):
    def __init__(self, time_step, units, channels, Ks, Kt, flag=0, device='cuda:0'):
        super(stemGNN_block, self).__init__()
        self.time_step = time_step
        self.units = units
        self.channels = channels
        self.Ks = Ks
        self.Kt = Kt
        self.flag = flag
        c_si, c_t, c_oo = self.channels
        self.temporal_conv_layer_input_InputConv2D = torch.nn.Conv2d(c_si, c_t, (1, 1), padding='same').to(device)
        self.temporal_conv_layer_input_Conv2D = torch.nn.Conv2d(c_si, c_t, (1, self.Kt), padding='valid').to(device)
        self.temporal_conv_layer_InputConv2D = torch.nn.Conv2d(c_si, c_t, (1, 1), padding='same').to(device)
        self.temporal_conv_layer_GLUConv2D = torch.nn.Conv2d(c_si, 2 * c_t, (1, self.Kt), padding='valid').to(device)
        self.temporal_conv_layer_Conv2D = torch.nn.Conv2d(c_si, c_t, (1, self.Kt), padding='valid').to(device)
        self.temporal_conv_layer_imag_InputConv2D = torch.nn.Conv2d(c_si, c_t, (1, 1), padding='same').to(device)
        self.temporal_conv_layer_imag_GLUConv2D = torch.nn.Conv2d(c_si, 2 * c_t, (1, self.Kt), padding='valid').to(
            device)
        self.temporal_conv_layer_imag_Conv2D = torch.nn.Conv2d(c_si, c_t, (1, self.Kt), padding='valid').to(device)
        self.spatio_conv_layer_fft_0221_InputConv2D = torch.nn.Conv2d(c_t, c_t, (1, 1), padding='same').to(device)
        self.ws = nn.Parameter(torch.zeros(size=(self.units, self.units)), requires_grad=True).to(device)
        nn.init.xavier_uniform_(self.ws.data, gain=1.414)
        self.bs = nn.Parameter(torch.zeros(size=(c_t, 1)), requires_grad=True).to(device)
        nn.init.xavier_uniform_(self.bs.data, gain=1.414)
        self.t_linear = torch.nn.Linear(c_t, c_t).to(device)
        self.temporal_conv_layer_output_InputConv2D = torch.nn.Conv2d(c_t, c_oo, (1, 1), padding='same').to(device)
        self.temporal_conv_layer_output_GLUConv2D = torch.nn.Conv2d(c_t, 2 * c_oo, (1, self.Kt), padding='valid').to(
            device)
        self.temporal_conv_layer_output_Conv2D = torch.nn.Conv2d(c_t, c_oo, (1, self.Kt), padding='valid').to(device)
        self.temporal_conv_layer_output_imag_InputConv2D = torch.nn.Conv2d(c_t, c_oo, (1, 1), padding='same').to(device)
        self.temporal_conv_layer_output_imag_GLUConv2D = torch.nn.Conv2d(c_t, 2 * c_oo, (1, self.Kt),
                                                                         padding='valid').to(
            device)
        self.temporal_conv_layer_output_imag_Conv2D = torch.nn.Conv2d(c_t, c_oo, (1, self.Kt), padding='valid').to(
            device)
        self.fore_auto_InputConv2D = torch.nn.Conv2d(c_si, c_t, (1, 1), padding='same').to(device)
        self.en_linear = nn.Linear(self.time_step - 4, self.time_step - 6)
        self.de_linear = nn.Linear(c_si, c_t)

    def temporal_conv_layer_input(self, x, Kt, c_in, c_out, device='cuda:0'):
        # print('Into tcli')
        _, T, n, _ = x.shape[:]

        if c_in > c_out:
            x_input = self.temporal_conv_layer_input_InputConv2D(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        elif c_in < c_out:
            x_input = torch.cat((x, torch.zeros([x.shape[0], T, n, c_out - c_in]).to(device)), dim=3)
        else:
            x_input = x

        _, time_step_temp, route_temp, channle_temp = x_input.shape[:]
        x_input = x_input[:, Kt - 1:T, :, :]

        x_conv = self.temporal_conv_layer_input_Conv2D(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        # return torch.relu(x_conv + x_input)
        return x_conv + x_input

    def temporal_conv_layer(self, x, Kt, c_in, c_out, act_func='relu', device='cuda:0'):
        _, T, n, _ = x.shape[:]

        if c_in > c_out:
            x_input = self.temporal_conv_layer_InputConv2D(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        elif c_in < c_out:
            x_input = torch.cat((x, torch.zeros([x.shape[0], T, n, c_out - c_in]).to(device)), dim=3)
        else:
            x_input = x

        _, time_step_temp, route_temp, channel_temp = x_input.shape[:]
        x_input = x_input[:, Kt - 1:T, :, :]

        if act_func == 'GLU':
            x_conv = self.temporal_conv_layer_GLUConv2D(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
            # return (x_conv[:, :, :, 0:c_out] + x_input) * torch.sigmoid(x_conv[:, :, :, -c_out:])
            return (x_conv[:, :, :, 0:c_out] + x_input) * x_conv[:, :, :, -c_out:]

        else:
            x_conv = self.temporal_conv_layer_Conv2D(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
            if act_func == 'sigmoid':
                # return torch.sigmoid(x_conv)
                return x_conv
            elif act_func == 'relu':
                # return torch.relu(x_conv + x_input)
                return x_conv + x_input

    def temporal_conv_layer_imag(self, x, Kt, c_in, c_out, act_func='relu', device='cuda:0'):
        _, T, n, _ = x.shape[:]

        if c_in > c_out:
            x_input = self.temporal_conv_layer_imag_InputConv2D(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        elif c_in < c_out:
            x_input = torch.cat((x, torch.zeros([x.shape[0], T, n, c_out - c_in]).to(device)), dim=3)
        else:
            x_input = x

        _, time_step_temp, route_temp, channel_temp = x_input.shape[:]
        x_input = x_input[:, Kt - 1:T, :, :]

        if act_func == 'GLU':
            x_conv = self.temporal_conv_layer_imag_GLUConv2D(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
            # return (x_conv[:, :, :, 0:c_out] + x_input) * torch.sigmoid(x_conv[:, :, :, -c_out:])
            return (x_conv[:, :, :, 0:c_out] + x_input) * x_conv[:, :, :, -c_out:]
        else:
            x_conv = self.temporal_conv_layer_imag_Conv2D(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
            # return torch.relu(x_conv + x_input)
            return x_conv + x_input

    def temporal_conv_layer_output(self, x, Kt, c_in, c_out, act_func='relu', device='cuda:0'):
        _, T, n, _ = x.shape[:]

        if c_in > c_out:
            x_input = self.temporal_conv_layer_output_InputConv2D(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        elif c_in < c_out:
            x_input = torch.cat((x, torch.zeros([x.shape[0], T, n, c_out - c_in]).to(device)), dim=3)
        else:
            x_input = x

        _, time_step_temp, route_temp, channel_temp = x_input.shape[:]
        x_input = x_input[:, Kt - 1:T, :, :]

        if act_func == 'GLU':
            x_conv = self.temporal_conv_layer_output_GLUConv2D(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
            # return (x_conv[:, :, :, 0:c_out] + x_input) * torch.sigmoid(x_conv[:, :, :, -c_out:])
            return (x_conv[:, :, :, 0:c_out] + x_input) * x_conv[:, :, :, -c_out:]

        else:
            x_conv = self.temporal_conv_layer_output_Conv2D(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
            if act_func == 'sigmoid':
                return torch.sigmoid(x_conv)
                # return x_conv
            elif act_func == 'relu':
                # return torch.relu(x_conv + x_input)
                return x_conv + x_input

    def temporal_conv_layer_imag_output(self, x, Kt, c_in, c_out, act_func='relu', device='cuda:0'):
        _, T, n, _ = x.shape[:]

        if c_in > c_out:
            x_input = self.temporal_conv_layer_output_imag_InputConv2D(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        elif c_in < c_out:
            x_input = torch.cat((x, torch.zeros([x.shape[0], T, n, c_out - c_in]).to(device)), dim=3)
        else:
            x_input = x

        _, time_step_temp, route_temp, channel_temp = x_input.shape[:]
        x_input = x_input[:, Kt - 1:T, :, :]

        if act_func == 'GLU':
            x_conv = self.temporal_conv_layer_output_imag_GLUConv2D(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
            # return (x_conv[:, :, :, 0:c_out] + x_input) * torch.sigmoid(x_conv[:, :, :, -c_out:])
            return (x_conv[:, :, :, 0:c_out] + x_input) * x_conv[:, :, :, -c_out:]
        else:
            x_conv = self.temporal_conv_layer_output_imag_Conv2D(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
            # return torch.relu(x_conv + x_input)
            return x_conv + x_input

    def spatio_conv_layer_fft_0221(self, x, Ks, c_in, c_out, e, device='cuda:0'):
        _, T, n, _ = x.shape[:]

        Ks = 1

        if c_in > c_out:
            x_input = self.spatio_conv_layer_fft_0221_InputConv2D(x)
        elif c_in < c_out:
            x_input = torch.cat((x, torch.zeros([x.shape[0], T, n, c_out - c_in]).to(device)), dim=3)
        else:
            x_input = x

        GF = x
        x_gc = self.gconv_fft_cnn_0221(GF, Ks, c_in, c_out, e) + self.bs.squeeze()

        x_gc = torch.reshape(x_gc, (-1, T, n, c_out))
        return x_gc

    def gconv_fft_cnn_0221(self, x, Ks, c_in, c_out, e, device='cuda:0'):
        kernel = e.unsqueeze(-1)
        kernel = torch.diag(e)
        n = kernel.shape[0]

        batch_size, time_step, n_route, c_in = x.shape[:]

        x_tmp = torch.reshape(x.permute(0, 1, 3, 2), (-1, n))
        real_kernel = torch.multiply(self.ws, kernel)
        x_mul = torch.matmul(x_tmp, real_kernel)

        x_gconv = torch.reshape(x_mul, (-1, c_out, n)).permute(0, 2, 1)
        return x_gconv

    def fc(self, x, device='cuda:0'):
        _, time_step_temp, route_temp, channel_temp = x.shape[:]
        x_tmp = torch.reshape(x, (-1, channel_temp))
        # hidden = torch.sigmoid(self.t_linear(x_tmp))
        hidden = self.t_linear(x_tmp)
        # out = torch.softmax(hidden, dim=-1)
        out = hidden
        outputs = torch.reshape(out, (-1, time_step_temp, route_temp, channel_temp))
        return outputs

    def fore_auto(self, x, Kt, c_in, c_out, device='cuda:0'):
        _, T, n, _ = x.shape[:]

        if c_in > c_out:
            x_input = self.fore_auto_InputConv2D(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        elif c_in < c_out:
            x_input = torch.cat((x, torch.zeros([x.shape[0], T, n, c_out - c_in]).to(device)), dim=3)
        else:
            x_input = x

        _, time_step_temp, route_temp, channel_temp = x.shape[:]
        x_input = x_input[:, Kt - 1:T, :, :]

        x_tmp = torch.reshape(x.permute(0, 2, 3, 1), (-1, time_step_temp))
        # hidden = torch.sigmoid(self.en_linear(x_tmp))
        hidden = self.en_linear(x_tmp)
        hidden = torch.reshape(torch.reshape(hidden, (-1, route_temp, channel_temp, T - Kt + 1)).permute(0, 3, 1, 2),
                               (-1, channel_temp))
        # out = torch.softmax(self.de_linear(hidden), dim=-1)
        out = self.de_linear(hidden)
        outputs = torch.reshape(out, (-1, T - Kt + 1, route_temp, c_out))
        return outputs

    def forward(self, x, Ks, Kt, channels, e, v, flag=0, back_forecast=None):
        # print('Into blok {}'.format(scope))
        c_si, c_t, c_oo = channels

        _, T, n, c_in = x.shape[:]

        x_input = self.temporal_conv_layer_input(x, Kt, c_si, c_t)

        if flag == 0:
            back_forecast = x_input
        else:
            back_forecast = self.fore_auto(back_forecast, Kt, c_in, c_t)

        GF = graph_fft(x, v, True)
        x = GF

        x = torch.fft.fft(x)
        x_real = x.real
        x_imag = x.imag
        x_real = self.temporal_conv_layer(x_real, Kt, c_si, c_t, 'GLU')
        x_imag = self.temporal_conv_layer_imag(x_imag, Kt, c_si, c_t, 'GLU')
        _, T, n, _ = x_real.shape[:]
        x_complex = torch.view_as_complex(torch.stack((x_real, x_imag), -1))
        x = torch.fft.ifft(x_complex).real

        _, _, _, c_fft = x.shape[:]
        x = self.spatio_conv_layer_fft_0221(x, Ks, c_fft, c_fft, e)

        GF = graph_fft(x, v, False)
        x = GF

        back_cast = self.fc(x)
        x = back_cast
        # x = torch.relu(x[:, :, :, 0:c_fft] + x_input)
        x = x[:, :, :, 0:c_fft] + x_input

        if flag == 0:
            # x = torch.relu(x[:, :, :, 0:c_fft] + x_input)
            x = x[:, :, :, 0:c_fft] + x_input
            # l1 = l2_loss(x_input - x[:, :, :, 0:c_fft])
        else:
            # x = torch.relu(x[:, :, :, 0:c_fft] + x_input + back_forecast)
            x = x[:, :, :, 0:c_fft] + x_input + back_forecast

        x_t = x

        x_o = self.temporal_conv_layer_output(x_t, Kt, c_t, c_oo)
        back_cast_o = self.temporal_conv_layer_imag_output(back_forecast, Kt, c_t, c_oo)

        x_ln = x_o

        fore_cast_ln = back_cast_o
        return x_ln, fore_cast_ln

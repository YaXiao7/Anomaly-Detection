# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in, d):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=0,
                                  stride=1
                                  )
        self.pad1 = nn.Conv1d(in_channels=c_in,
                              out_channels=c_in,
                              kernel_size=1,
                              padding=0,
                              stride=1
                              )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.d = d
        self.dropout = nn.Dropout(0.1)
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # print(self.d)
        if self.d == 1:
            x_i = x.clone()
            x_p1 = self.downConv(x.permute(0, 2, 1))
            x_p2 = self.pad1(x_i[:, 0:2, :].permute(0, 2, 1))
            x_p = torch.cat((x_p1, x_p2), 2)
            x = self.norm(x_p)
            x = self.dropout(self.activation(x))
            x = x + x_i.permute(0, 2, 1)
            x = self.maxPool(x)
            x = x.transpose(1, 2)
            return x
        elif self.d == 2:
            x_i = x.clone()
            x_p = x.permute(0, 2, 1)
            x1 = x[:, 0::2, :]
            x1_p1 = self.downConv(x1.permute(0, 2, 1))
            x1_p2 = self.pad1(x1[:, 0:2, :].permute(0, 2, 1))
            x1_p = torch.cat((x1_p1, x1_p2), 2)
            x2 = x[:, 1::2, :]
            x2_p1 = self.downConv(x2.permute(0, 2, 1))
            x2_p2 = self.pad1(x2[:, 0:2, :].permute(0, 2, 1))
            x2_p = torch.cat((x2_p1, x2_p2), 2)
            for i in range(x_p.shape[2]):
                if i % 2 == 0:
                    x_p[:, :, i] = x1_p[:, :, i // 2]
                else:
                    x_p[:, :, i] = x2_p[:, :, i // 2]
            x = self.norm(x_p)
            x = self.dropout(self.activation(x))
            x = x + x_i.permute(0, 2, 1)
            x = self.maxPool(x)
            x = x.transpose(1, 2)
            return x
        else:
            x_i = x.clone()
            x_p = x.permute(0, 2, 1)
            for i in range(self.d):
                x1 = x[:, i::self.d,:]
                x1_p1 = self.downConv(x1.permute(0, 2, 1))
                x1_p2 = self.pad1(x1[:, 0:2, :].permute(0, 2, 1))
                x1_p = torch.cat((x1_p1, x1_p2), 2)
                for j in range(x_p.shape[2]):
                    if j % self.d == i:
                        x_p[:, :, j] = x1_p[:, :, j // self.d]
            x = self.norm(x_p)
            x = self.dropout(self.activation(x))
            x = x + x_i.permute(0, 2, 1)
            x = self.maxPool(x)
            x = x.transpose(1, 2)
            return x

class NonParametricFourierLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NonParametricFourierLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight_real = nn.Parameter(torch.randn(output_dim, input_dim))
        self.weight_imag = nn.Parameter(torch.randn(output_dim, input_dim))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Reshape input tensor for matrix multiplication
        x = x.view(-1, self.input_dim)

        # Compute Fourier transform
        real_part = torch.matmul(x, self.weight_real.t())
        imag_part = torch.matmul(x, self.weight_imag.t())

        # Reshape output tensor back to original shape
        real_part = real_part.view(batch_size, seq_len, self.output_dim)
        imag_part = imag_part.view(batch_size, seq_len, self.output_dim)

        # Combine real and imaginary parts
        fourier_result = torch.cat((real_part, imag_part), dim=-1)

        return fourier_result

class NormFourier(nn.Module):
    def __init__(self, input_dim):
        super(NormFourier, self).__init__()
        self.input_dim = input_dim

        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        norm_x = self.norm(x)
        return norm_x

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", ECSP=False , EFOURIER=False):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=d_model // 2, out_channels=d_model // 2, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model // 2) if ECSP else nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.csp = ECSP
        self.fourier = EFOURIER


    def forward(self, x, attn_mask=None):
        # x [B, L, D]

        # split in half
        if self.csp:
            # print('using csp')
            split_x = torch.split(x, x.shape[2] // 2, dim=2)
            csp_x = split_x[1].clone()
            norm_x = split_x[0].clone()

            if self.fourier:
                split_csp_x = torch.split(csp_x, csp_x.shape[2] // 2, dim=2)
                four_x = split_csp_x[0].clone()
                cspp_x = split_csp_x[1].clone()

                # Apply non-parametric Fourier layer on four_x
                four_x = self.non_parametric_fourier(four_x)
                four_x = self.norm_fourier(four_x)

                # Self-attention and residual connection on cspp_x
                new_x, attn = self.attention(
                    cspp_x, cspp_x, cspp_x,
                    attn_mask=attn_mask
                )
                cspp_x = cspp_x + self.dropout(new_x)
                cspp_x = self.norm1(cspp_x)

                # Concatenate cspp_x, four_x, and norm_x along feature dimension
                x = torch.cat((cspp_x, four_x, norm_x), 2)
            else:
                norm_x = self.conv3(norm_x.permute(0, 2, 1))
                norm_x = norm_x.transpose(1, 2)

                new_x, attn = self.attention(
                    csp_x, csp_x, csp_x,
                    attn_mask=attn_mask
                )
                csp_x = csp_x + self.dropout(new_x)
                csp_x = self.norm1(csp_x)

                # Concatenate csp_x and norm_x along feature dimension
                x = torch.cat((csp_x, norm_x), 2)

            y = x
        else:
            # print('not using csp')
            new_x, attn = self.attention(
                x, x, x,
                attn_mask=attn_mask
            )
            x = x + self.dropout(new_x)
            y = x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class FocusLayer(nn.Module):
    # Focus l information into d-space
    def __init__(self, c1, c2, k=1):
        super(FocusLayer, self).__init__()
        # self.conv = nn.Conv1d(in_channels=c1*2, out_channels=c2, kernel_size=1)

    def forward(self, x):  # x(b,d,l) -> y(b,2d,l/2)
        return torch.cat([x[..., ::2], x[..., 1::2]], dim=1)


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, Focus_layer=None, Passthrough_layer=None):
        super(Encoder, self).__init__()
        self.passnum = len(attn_layers)
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.f_F = Focus_layer
        self.passthrough = Passthrough_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        x_out_list = []
        if self.conv_layers is not None:
            if self.f_F is not None:
                # print('using focus')
                i = self.passnum
                for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                    x, attn = attn_layer(x, attn_mask=attn_mask)
                    i = i - 1
                    x_out = (x.clone()).permute(0, 2, 1)
                    for _ in range(i):
                        x_out = self.f_F(x_out)
                    x_out_list.append(x_out.transpose(1, 2))
                    x = conv_layer(x)
                    attns.append(attn)
                x, attn = self.attn_layers[-1](x)
                x_out_list.append(x)
                attns.append(attn)
            else:
                for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                    x, attn = attn_layer(x, attn_mask=attn_mask)
                    x = conv_layer(x)
                    attns.append(attn)
                x, attn = self.attn_layers[-1](x)
                attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if (self.passthrough is not None) and (self.conv_layers is not None):
            # print('using passthrough')
            x_pass = torch.cat(x_out_list, -1)
            x_pass = x_pass.permute(0, 2, 1)
            x_final = self.passthrough(x_pass)
            x = x_final.transpose(1, 2)


        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = [];
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s);
            attns.append(attn)
        x_stack = torch.cat(x_stack, -2)

        return x_stack, attns

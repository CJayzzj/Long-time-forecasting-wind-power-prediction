"""
LMTF-EIA-DyT with configurable multi-scale kernels (kernel_nums / kernel_sizes).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from layers.RevIN import RevIN


def SinCosPosEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3):
    x = 0.5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = (
            2
            * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x)
            * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x)
            - 1
        )
        if abs(cpe.mean()) <= eps:
            break
        if cpe.mean() > eps:
            x += 0.001
        else:
            x -= 0.001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (0.5 if exponential else 1)) - 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    if pe is None or pe == 'no':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe in ('normal', 'gauss'):
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos':
        W_pos = SinCosPosEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(f"{pe} is not a valid pe")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


class DyT(nn.Module):
    def __init__(self, C, init_alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(C))
        self.beta = nn.Parameter(torch.zeros(C))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return self.gamma * x + self.beta


class DataEmbedding(nn.Module):
    def __init__(self, pe_type, seq_len, d_model, c_in, dropout=0.0):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.position_embedding = positional_encoding(pe=pe_type, learn_pe=True, q_len=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding
        return self.dropout(x)


class LD(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            stride=1,
            padding=int(kernel_size // 2),
            padding_mode='replicate',
            bias=True,
        )
        kernel_size_half = kernel_size // 2
        sigma = 1.0
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            weights[0, 0, i] = math.exp(-((i - kernel_size_half) / (2 * sigma)) ** 2)
        self.conv.weight.data = F.softmax(weights, dim=-1)
        self.conv.bias.data.fill_(0.0)

    def forward(self, x):
        B, D, N = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(B * N, 1, D)
        trend = self.conv(x)
        trend = trend.reshape(B, N, D)
        trend = trend.permute(0, 2, 1)
        return trend


class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)

    def forward(self, x):
        return self.avg(x)


class EventIntensityDetector(nn.Module):
    def __init__(self, d_model, num_variables, dropout=0.1):
        super().__init__()
        self.ot_encoder = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model * 2, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=d_model * 2, out_channels=d_model, kernel_size=1),
            nn.ReLU(),
        )
        self.intensity_projection = nn.Conv1d(in_channels=1, out_channels=num_variables, kernel_size=1)

    def forward(self, ot_tensor):
        ot_encoded = self.ot_encoder(ot_tensor)
        ot_reshaped = ot_encoded.permute(0, 2, 1)
        intensity = self.intensity_projection(ot_reshaped)
        intensity = intensity.permute(0, 2, 1)
        return intensity


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, num_variables, n_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = DyT(d_model, init_alpha=0.5)
        self.norm2 = DyT(d_model, init_alpha=0.5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        q_reshaped = q.permute(0, 2, 1)
        k_reshaped = k.permute(0, 2, 1)
        v_reshaped = v.permute(0, 2, 1)

        residual = q_reshaped
        attn_output, _ = self.self_attn(q_reshaped, k_reshaped, v_reshaped)
        output = self.norm1(residual + self.dropout(attn_output))

        ffn_output = self.ffn(output)
        output = self.norm2(output + self.dropout(ffn_output))
        output = output.permute(0, 2, 1)
        return output


class MultiScaleMovingAverage(nn.Module):
    def __init__(self, kernel_sizes, d_model=512, num_variables=10):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)
        self.moving_avg_layers = nn.ModuleList([MovingAvg(kernel_size=k, stride=1) for k in kernel_sizes])
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        self.output_projection = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)

    def forward(self, x):
        scale_outputs = [layer(x) for layer in self.moving_avg_layers]
        weighted_output = 0
        for i, output in enumerate(scale_outputs):
            weighted_output += self.scale_weights[i] * output
        return self.output_projection(weighted_output)


def parse_kernel_sizes(kernel_sizes_str):
    if kernel_sizes_str is None or kernel_sizes_str == 'default':
        return [25, 13, 7, 5, 3]
    return [int(k) for k in str(kernel_sizes_str).split('_')]


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.revin = configs.revin
        self.revin_layer = RevIN(channel=configs.enc_in, output_dim=configs.pred_len)

        self.position_embedder = DataEmbedding(
            pe_type=configs.pe_type,
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            c_in=configs.enc_in,
            dropout=configs.dropout,
        )

        self.LD = LD(kernel_size=25)

        kernel_sizes = parse_kernel_sizes(getattr(configs, 'kernel_sizes', 'default'))
        self.kernel_sizes = kernel_sizes
        self.trend_branch = MultiScaleMovingAverage(
            kernel_sizes=kernel_sizes,
            d_model=configs.d_model,
            num_variables=configs.enc_in,
        )

        self.event_detector = EventIntensityDetector(
            d_model=configs.d_model,
            num_variables=configs.enc_in,
            dropout=configs.dropout,
        )

        self.n_layers = configs.n_layers
        self.residual_attn_blocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    d_model=configs.d_model,
                    num_variables=configs.enc_in,
                    n_heads=configs.n_heads,
                    dropout=configs.dropout,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.Linear_trend = nn.Linear(configs.d_model, configs.pred_len)
        self.Linear_res = nn.Linear(configs.d_model, configs.pred_len)
        self.Linear_trend.weight = nn.Parameter((1 / configs.d_model) * torch.ones([configs.pred_len, configs.d_model]))
        self.Linear_res.weight = nn.Parameter((1 / configs.d_model) * torch.ones([configs.pred_len, configs.d_model]))

        print(f"[LMTF-EIA-DyT-kernel] kernel_sizes: {kernel_sizes}")

    def forward(self, x):
        B, S, N = x.shape
        if self.revin:
            x = self.revin_layer(x)

        inp = self.position_embedder(x.permute(0, 2, 1))
        inp = inp.permute(0, 2, 1)

        main = self.LD(inp)
        residual = inp - main

        trend_output = self.trend_branch(main) + main

        ot_tensor = inp[:, :, -1:]
        event_intensity = self.event_detector(ot_tensor)

        res_attn = residual
        for i in range(self.n_layers):
            res_attn = self.residual_attn_blocks[i](q=res_attn, k=event_intensity, v=event_intensity)

        trend_pred = self.Linear_trend(trend_output.permute(0, 2, 1)).permute(0, 2, 1)
        residual_pred = self.Linear_res(res_attn.permute(0, 2, 1)).permute(0, 2, 1)

        pred_all = trend_pred + residual_pred
        if self.revin:
            pred_all = self.revin_layer.inverse_normalize(pred_all)
        return pred_all


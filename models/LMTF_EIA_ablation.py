"""
LMTF-EIA模型消融实验变体
用于验证Innovation 1（EIA机制）各组件的贡献

消融实验设计（聚焦EIA机制的层次化消融）：
  模块级消融:
    1. LMTF-EIA_woEIA: 去掉整个EIA机制，残差直接投影 → 验证Innovation 1整体
  组件级消融:
    2. LMTF-EIA_woEID: 去掉事件强度检测，改用自注意力 → 验证EID组件
    3. LMTF-EIA_woResAttn: 去掉注意力块，改用MLP     → 验证注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import MultiheadAttention
from layers.RevIN import RevIN

# ============== 从LMTF_EIA.py复制的基础模块 ==============

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
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def positional_encoding(pe, learn_pe, q_len, d_model):
    if pe == None or pe == 'no':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = SinCosPosEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe")
    return nn.Parameter(W_pos, requires_grad=learn_pe)

class DataEmbedding(nn.Module):
    def __init__(self, pe_type, seq_len, d_model, c_in, dropout=0.):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.position_embedding = positional_encoding(pe=pe_type, learn_pe=True, q_len=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding
        return self.dropout(x)

class LD(nn.Module):
    """可学习分解模块"""
    def __init__(self, kernel_size=25):
        super(LD, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, 
                             padding=int(kernel_size//2), padding_mode='replicate', bias=True)
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
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        
    def forward(self, x):
        x_pooled = self.avg(x)
        return x_pooled


class EventIntensityDetector(nn.Module):
    def __init__(self, d_model, num_variables, dropout=0.1):
        super(EventIntensityDetector, self).__init__()
        self.ot_encoder = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model*2, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=d_model*2, out_channels=d_model, kernel_size=1),
            nn.ReLU()
        )
        # 保持原始设计：in_channels=1, out_channels=num_variables
        # 这样可以从OT单点信息扩展为多变量信息
        self.intensity_projection = nn.Conv1d(
            in_channels=1,
            out_channels=num_variables,
            kernel_size=1
        )
        
    def forward(self, ot_tensor):
        # ot_tensor: (B, D, 1) - OT变量的单点表示
        B, D, _ = ot_tensor.shape
        ot_encoded = self.ot_encoder(ot_tensor)  # (B, D, 1)
        # 重新排列: (B, D, 1) -> (B, 1, D)
        ot_reshaped = ot_encoded.permute(0, 2, 1)  # (B, 1, D)
        # 投影到 num_variables
        intensity = self.intensity_projection(ot_reshaped)  # (B, num_vars, D)
        # 转换回: (B, num_vars, D) -> (B, D, num_vars)
        intensity = intensity.permute(0, 2, 1)  # (B, D, num_vars)
        return intensity

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, num_variables, n_heads=8, dropout=0.1):
        super(ResidualAttentionBlock, self).__init__()
        self.self_attn = MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        B, D, N = q.shape
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
    def __init__(self, kernel_sizes=None, d_model=512, num_variables=10):
        super(MultiScaleMovingAverage, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [25, 13, 7, 5, 3]
            
        self.moving_avg_layers = nn.ModuleList([
            MovingAvg(kernel_size=k, stride=1) for k in kernel_sizes
        ])
        self.scale_weights = nn.Parameter(torch.ones(len(kernel_sizes)) / len(kernel_sizes))
        self.output_projection = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=1
        )
        
    def forward(self, x):
        scale_outputs = []
        for layer in self.moving_avg_layers:
            scale_out = layer(x)
            scale_outputs.append(scale_out)
        
        weighted_output = 0
        for i, output in enumerate(scale_outputs):
            weighted_output += self.scale_weights[i] * output
        
        output = self.output_projection(weighted_output)
        return output


# ============== 消融实验模型变体 ==============

class Model_woEventIntensity(nn.Module):
    """
    消融实验1: 去掉事件强度检测，使用自注意力
    验证事件强度检测模块的贡献
    """
    def __init__(self, configs):
        super(Model_woEventIntensity, self).__init__()
        self.configs = configs
        
        self.revin = configs.revin
        self.revin_layer = RevIN(channel=configs.enc_in, output_dim=configs.pred_len)
        
        self.position_embedder = DataEmbedding(
            pe_type=configs.pe_type,
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            c_in=configs.enc_in,
            dropout=configs.dropout
        )
        
        self.LD = LD(kernel_size=25)
        
        self.trend_branch = MultiScaleMovingAverage(
            kernel_sizes=[25, 13, 7, 5, 3],
            d_model=configs.d_model,
            num_variables=configs.enc_in
        )
        
        # 不使用事件强度检测，使用自注意力
        self.n_layers = configs.n_layers
        self.self_attn_blocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=configs.d_model,
                num_variables=configs.enc_in,
                n_heads=configs.n_heads,
                dropout=configs.dropout
            ) for _ in range(self.n_layers)
        ])
        
        self.Linear_trend = nn.Linear(configs.d_model, configs.pred_len)
        self.Linear_res = nn.Linear(configs.d_model, configs.pred_len)
        self.Linear_trend.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([configs.pred_len, configs.d_model])) 
        self.Linear_res.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([configs.pred_len, configs.d_model])) 
        
    def forward(self, x):
        B, S, N = x.shape
        if self.revin:
            x = self.revin_layer(x)
            
        inp = self.position_embedder(x.permute(0, 2, 1))
        inp = inp.permute(0, 2, 1)
        
        main = self.LD(inp)
        residual = inp - main
        
        trend_output = self.trend_branch(main) + main
        
        # 使用自注意力（Q=K=V=残差），而不是Cross-Attention
        res_attn = residual
        for i in range(self.n_layers):
            res_attn = self.self_attn_blocks[i](
                q=res_attn,
                k=res_attn,  # K=残差本身
                v=res_attn   # V=残差本身
            )
        
        residual_output = res_attn
        
        trend_pred = self.Linear_trend(trend_output.permute(0,2,1)).permute(0,2,1)
        residual_pred = self.Linear_res(residual_output.permute(0,2,1)).permute(0,2,1)
        
        pred_all = trend_pred + residual_pred
        if self.revin:
            pred_all = self.revin_layer.inverse_normalize(pred_all)
        return pred_all


class Model_woResAttn(nn.Module):
    """
    消融实验2: 去掉注意力机制，保留事件强度检测（EID）
    设计模式：ot_tensor -> EID -> Linear_res（不经过注意力）
    用于验证：没有ResAttn时，EID分支本身的贡献
    """
    def __init__(self, configs):
        super(Model_woResAttn, self).__init__()
        self.configs = configs
        
        self.revin = configs.revin
        self.revin_layer = RevIN(channel=configs.enc_in, output_dim=configs.pred_len)
        
        self.position_embedder = DataEmbedding(
            pe_type=configs.pe_type,
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            c_in=configs.enc_in,
            dropout=configs.dropout
        )
        
        self.LD = LD(kernel_size=25)
        
        self.trend_branch = MultiScaleMovingAverage(
            kernel_sizes=[25, 13, 7, 5, 3],
            d_model=configs.d_model,
            num_variables=configs.enc_in
        )

        # 保留EID模块，去掉注意力模块
        self.event_detector = EventIntensityDetector(
            d_model=configs.d_model,
            num_variables=configs.enc_in,
            dropout=configs.dropout
        )
        
        self.Linear_trend = nn.Linear(configs.d_model, configs.pred_len)
        self.Linear_res = nn.Linear(configs.d_model, configs.pred_len)
        self.Linear_trend.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([configs.pred_len, configs.d_model])) 
        self.Linear_res.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([configs.pred_len, configs.d_model])) 
        
    def forward(self, x):
        B, S, N = x.shape
        if self.revin:
            x = self.revin_layer(x)
            
        inp = self.position_embedder(x.permute(0, 2, 1))
        inp = inp.permute(0, 2, 1)
        
        main = self.LD(inp)
        residual = inp - main #（B, D, N）
        
        trend_output = self.trend_branch(main) + main
        
        ot_tensor = inp[:, :, -1:]  # (B, D, 1) 提取OT变量
        
        # 事件强度检测
        event_intensity = self.event_detector(ot_tensor)  # (B, D, 1) -> (B, D, N)

        # 去掉注意力后，EID输出直接进入线性预测头。
        # 这里显式统一成 (B, N, D) 再送入 Linear_res(D -> P)，
        # 避免远端环境中张量排布差异导致 matmul 维度错误。
        residual_output = event_intensity

        trend_pred = self.Linear_trend(trend_output.permute(0, 2, 1)).permute(0, 2, 1)

        residual_features = residual_output
        if residual_features.shape[-1] != self.configs.d_model:
            if residual_features.shape[1] == self.configs.d_model:
                residual_features = residual_features.permute(0, 2, 1)
            else:
                raise RuntimeError(
                    f"Unexpected woResAttn residual feature shape: {tuple(residual_features.shape)}; "
                    f"expected one dimension to equal d_model={self.configs.d_model}"
                )

        residual_pred = self.Linear_res(residual_features).permute(0, 2, 1)
        
        pred_all = trend_pred + residual_pred
        if self.revin:
            pred_all = self.revin_layer.inverse_normalize(pred_all)
        return pred_all


class Model_woEIA(nn.Module):
    """
    消融实验3: 去掉整个EIA机制（EventIntensityDetector + ResidualAttentionBlock）
    残差分量不经过任何注意力处理，直接投影到预测长度
    验证Innovation 1整体（事件强度驱动的残差注意力机制）的贡献
    """
    def __init__(self, configs):
        super(Model_woEIA, self).__init__()
        self.configs = configs
        
        self.revin = configs.revin
        self.revin_layer = RevIN(channel=configs.enc_in, output_dim=configs.pred_len)
        
        self.position_embedder = DataEmbedding(
            pe_type=configs.pe_type,
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            c_in=configs.enc_in,
            dropout=configs.dropout
        )
        
        self.LD = LD(kernel_size=25)
        
        self.trend_branch = MultiScaleMovingAverage(
            kernel_sizes=[25, 13, 7, 5, 3],
            d_model=configs.d_model,
            num_variables=configs.enc_in
        )
        
        # 不使用EventIntensityDetector和ResidualAttentionBlock
        # 残差直接通过输出投影层
        
        self.Linear_trend = nn.Linear(configs.d_model, configs.pred_len)
        self.Linear_res = nn.Linear(configs.d_model, configs.pred_len)
        self.Linear_trend.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([configs.pred_len, configs.d_model])) 
        self.Linear_res.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([configs.pred_len, configs.d_model])) 
        
    def forward(self, x):
        B, S, N = x.shape
        if self.revin:
            x = self.revin_layer(x)
            
        inp = self.position_embedder(x.permute(0, 2, 1))
        inp = inp.permute(0, 2, 1)
        
        main = self.LD(inp)
        residual = inp - main
        
        trend_output = self.trend_branch(main) + main
        
        # 残差不经过任何注意力处理，直接作为输出
        residual_output = residual
        
        trend_pred = self.Linear_trend(trend_output.permute(0,2,1)).permute(0,2,1)
        residual_pred = self.Linear_res(residual_output.permute(0,2,1)).permute(0,2,1)
        
        pred_all = trend_pred + residual_pred
        if self.revin:
            pred_all = self.revin_layer.inverse_normalize(pred_all)
        return pred_all


# ============== 统一的Model类，通过ablation_type参数选择变体 ==============

class Model(nn.Module):
    """
    消融实验统一入口 - 正交设计的消融矩阵
    通过configs.ablation_type选择不同的消融变体:
    
    消融矩阵：
    ├─ woEIA: 去掉EID和注意力，残差直接投影 → 基准线
    ├─ woResAttn: 去掉注意力，保留EID → 测试EID的贡献
    ├─ woEID: 去掉EID，用自注意力 → 测试注意力的贡献
    └─ full: 保留EID和注意力 → 完整模型
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # 获取消融类型，默认为完整模型
        ablation_type = getattr(configs, 'ablation_type', 'full')
        print(f"[LMTF-EIA_ablation] ablation_type: {ablation_type}")
        
        if ablation_type in ('woEID', 'woEventIntensity'):
            self.model = Model_woEventIntensity(configs)
        elif ablation_type == 'woResAttn':
            self.model = Model_woResAttn(configs)
        elif ablation_type == 'woEIA':
            self.model = Model_woEIA(configs)
        else:
            # 默认使用完整模型（从LMTF_EIA.py导入）
            from models.LMTF_EIA import Model as FullModel
            self.model = FullModel(configs)
    
    def forward(self, x):
        return self.model(x)

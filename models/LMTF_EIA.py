import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import MultiheadAttention
from layers.RevIN import RevIN

# 位置编码相关函数（保持不变）
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

# 数据嵌入层
class DataEmbedding(nn.Module):
    def __init__(self, pe_type, seq_len, d_model, c_in, dropout=0.):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)  # 从序列长度映射到d_model维度
        self.position_embedding = positional_encoding(pe=pe_type, learn_pe=True, q_len=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # x: (B, N, S) -> 经过线性层 -> (B, N, D) -> 加位置编码
        x = self.value_embedding(x) + self.position_embedding
        return self.dropout(x)

# 可学习分解模块
class LD(nn.Module):
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
        # x: (B, D, N) -> 对每个通道单独处理
        B, D, N = x.shape
        x = x.permute(0, 2, 1)  # (B, D, N) -> (B, N, D)
        x = x.reshape(B * N, 1, D)  # (B*N, 1, D)
        trend = self.conv(x)  # (B*N, 1, D) -> 卷积 -> (B*N, 1, D)
        trend = trend.reshape(B, N, D)  # (B*N, 1, D) -> (B, N, D)
        trend = trend.permute(0, 2, 1)  # (B, N, D) -> (B, D, N)
        return trend

# 移动平均模块
class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    修复了维度问题，确保输入输出形状一致
    """
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        # 使用AvgPool1d来处理(B, D, N)的形状
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        
    def forward(self, x):
        # x: (B, D, N)
        x_pooled = self.avg(x)  # (B, D, N) -> 池化 -> (B, D, N)
        return x_pooled

# 修复的事件强度检测模块
class EventIntensityDetector(nn.Module):
    def __init__(self, d_model, num_variables, dropout=0.1):
        super(EventIntensityDetector, self).__init__()
        # 使用Conv1d来处理(B, D, 1) -> (B, D, N)的映射
        self.ot_encoder = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model*2, kernel_size=1),  # (B, D, 1) -> (B, D*2, 1)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=d_model*2, out_channels=d_model, kernel_size=1),  # (B, D*2, 1) -> (B, D, 1)
            nn.ReLU()
        )
        # 使用Conv1d将1个通道扩展到num_variables个通道
        self.intensity_projection = nn.Conv1d(
            in_channels=1,  # 输入通道数为1
            out_channels=num_variables,  # 输出通道数为num_variables
            kernel_size=1
        )  # 从(B, 1, D) -> (B, N, D)
        
    def forward(self, ot_tensor):
        # ot_tensor: (B, D, 1) -> OT变量
        B, D, _ = ot_tensor.shape
        
        # 先进行编码
        ot_encoded = self.ot_encoder(ot_tensor)  # (B, D, 1) -> (B, D, 1)
        
        # 重新排列维度: (B, D, 1) -> (B, 1, D)
        ot_reshaped = ot_encoded.permute(0, 2, 1)  # (B, D, 1) -> (B, 1, D)
        
        # 投影到所有变量
        intensity = self.intensity_projection(ot_reshaped)  # (B, 1, D) -> (B, N, D)
        
        # 重新排列回原始维度: (B, N, D) -> (B, D, N)
        intensity = intensity.permute(0, 2, 1)  # (B, N, D) -> (B, D, N)
        
        return intensity  # (B, D, N) 作为K和V的生成

# 残差注意力模块
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
        # q, k, v: (B, D, N) -> 转换为(B, D, N)用于注意力计算
        B, D, N = q.shape
        
        # 转换维度以适应注意力机制：(B, D, N) -> (B, N, D)
        q_reshaped = q.permute(0, 2, 1)  # (B, D, N) -> (B, N, D)
        k_reshaped = k.permute(0, 2, 1)  # (B, D, N) -> (B, N, D)
        v_reshaped = v.permute(0, 2, 1)  # (B, D, N) -> (B, N, D)
        
        # 第一次注意力计算
        residual = q_reshaped
        attn_output, _ = self.self_attn(q_reshaped, k_reshaped, v_reshaped)  # (B, N, D) -> (B, N, D)
        output = self.norm1(residual + self.dropout(attn_output))  # 残差连接+归一化
        
        # Feed Forward Network
        ffn_output = self.ffn(output)  # (B, N, D) -> (B, N, D)
        output = self.norm2(output + self.dropout(ffn_output))  # 残差连接+归一化
        
        # 转换回原始维度
        output = output.permute(0, 2, 1)  # (B, N, D) -> (B, D, N)
        
        return output  # (B, D, N)

# 多尺度移动平均集成模块
class MultiScaleMovingAverage(nn.Module):
    def __init__(self, kernel_sizes=None, d_model=512, num_variables=10):
        super(MultiScaleMovingAverage, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [25, 13, 7, 5, 3]  # 多尺度卷积核大小
            
        self.moving_avg_layers = nn.ModuleList([
            MovingAvg(kernel_size=k, stride=1) for k in kernel_sizes
        ])
        
        # 可学习的权重用于集成不同尺度的输出
        self.scale_weights = nn.Parameter(torch.ones(len(kernel_sizes)) / len(kernel_sizes))
        
        # 输出投影层 - 使用1x1卷积
        self.output_projection = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=1
        )
        
    def forward(self, x):
        # x: (B, D, N) -> 趋势分量
        scale_outputs = []
        for layer in self.moving_avg_layers:
            scale_out = layer(x)  # 每个尺度: (B, D, N)
            scale_outputs.append(scale_out)
        
        # 多尺度集成 (加权平均)
        weighted_output = 0
        for i, output in enumerate(scale_outputs):
            weighted_output += self.scale_weights[i] * output
        
        # 使用1x1卷积进行投影
        output = self.output_projection(weighted_output)  # (B, D, N) -> (B, D, N)
        return output

# 主模型
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs

        self.revin=configs.revin # 是否启用RevIN，标准化输入
        self.revin_layer=RevIN(channel=configs.enc_in,output_dim=configs.pred_len) # RevIN用于标准化和逆标准化，接收输入通道configs.enc_in和输出维度configs.pred_len
        
        # 1. 位置编码
        self.position_embedder = DataEmbedding(
            pe_type=configs.pe_type,
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            c_in=configs.enc_in,
            dropout=configs.dropout
        )
        
        # 2. 可学习分解模块
        self.LD = LD(kernel_size=25)
        
        # 3. 趋势支路: 多尺度移动平均
        self.trend_branch = MultiScaleMovingAverage(
            kernel_sizes=[25, 13, 7, 5, 3],  # 多尺度卷积核
            d_model=configs.d_model,
            num_variables=configs.enc_in
        )
        
        # 4. 残差支路组件
        # 事件强度检测模块
        self.event_detector = EventIntensityDetector(
            d_model=configs.d_model,
            num_variables=configs.enc_in,
            dropout=configs.dropout
        )
        
        # 多层的残差注意力模块
        self.n_layers = configs.n_layers
        self.residual_attn_blocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=configs.d_model,
                num_variables=configs.enc_in,
                n_heads=configs.n_heads,
                dropout=configs.dropout
            ) for _ in range(self.n_layers)
        ])
        
        # 5. 输出投影层
        self.Linear_trend = nn.Linear(configs.d_model, configs.pred_len) # 线性层，用于处理Leddam模块的Trend分量，生成左支路预测值。
        self.Linear_res = nn.Linear(configs.d_model, configs.pred_len) # 线性层，用于处理Leddam模块的res分量，生成右支路预测值。

        self.Linear_trend.weight = nn.Parameter( # 设置self.Linear_trend和self.Linear_res的权重为常数(1 / configs.d_model)。使用torch.ones初始化权重矩阵。
                (1 / configs.d_model) * torch.ones([configs.pred_len, configs.d_model])) 
        self.Linear_res.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([configs.pred_len, configs.d_model])) 
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: (B, S, N) 输入序列
        Returns:
            pred: (B, pred_len, N)或者(B, pred_len, 1) 预测的风功率
        """
        B, S, N = x.shape
        if self.revin:
            x = self.revin_layer(x)
        # 1. 位置编码
        inp = self.position_embedder(x.permute(0, 2, 1))  # (B, N, S) -> (B, N, D)
        inp = inp.permute(0, 2, 1)  # (B, N, D) -> (B, D, N)
        
        # 2. 可学习分解
        main = self.LD(inp)  # 趋势分量: (B, D, N)
        residual = inp - main  # 残差分量: (B, D, N)
        
        # 3. 趋势支路处理
        trend_output = self.trend_branch(main) + main  # 多尺度移动平均: (B, D, N) -> (B, D, N)
        
        # 4. 残差支路处理
        # 提取OT变量 (最后一个变量)
        ot_tensor = inp[:, :, -1:]  # (B, D, 1) 提取OT变量
        
        # 事件强度检测
        event_intensity = self.event_detector(ot_tensor)  # (B, D, 1) -> (B, D, N)
        
        # 多层注意力处理
        res_attn = residual
        for i in range(self.n_layers):
            res_attn = self.residual_attn_blocks[i](
                q=res_attn,  # Q: 残差本身
                k=event_intensity,  # K: 事件强度
                v=event_intensity  # V: 事件强度
            )  # (B, D, N) -> (B, D, N)
        
        residual_output = res_attn  # 残差支路输出: (B, D, N)
        
        # 5. 输出投影
        # 使用Linear处理，维度不变
        trend_pred = self.Linear_trend(trend_output.permute(0,2,1)).permute(0,2,1)   # (B, D, N) -> (B, pred_len, N)
        residual_pred = self.Linear_res(residual_output.permute(0,2,1)).permute(0,2,1)  # (B, D, N) -> (B, pred_len, N)

        # 6. 合并趋势和残差
        pred_all = trend_pred + residual_pred  # (B, pred_len, N)
        if self.revin: # 如果使用了RevIN，则将预测结果pred通过inverse_normalize方法逆标准化，还原为原始尺度。
            pred_all=self.revin_layer.inverse_normalize(pred_all)
        return pred_all

        # # 7. 只输出OT变量 (多对单)
        # pred_ot = pred_all[:, :, -1:]  # (B, pred_len, 1) 只取最后一个变量(OT)
        # return pred_ot



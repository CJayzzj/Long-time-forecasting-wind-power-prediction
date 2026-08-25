import torch
import torch.nn as nn
import math
from torch import Tensor
import torch.nn.functional as F
from typing import  Optional

class Leddam(nn.Module):
    
    def __init__(self,
                 enc_in, # (输入)特征维度
                 seq_len, # (输入)序列长度
                 d_model, # 隐藏层维度
                 dropout, # dropout率
                 pe_type, # 位置编码类型
                 kernel_size, # 卷积核大小
                 n_layers=3): # 层数
        
        super(Leddam, self).__init__()
        self.n_layers=n_layers
        self.LD=LD(kernel_size=kernel_size) # 初始化LD模块(卷积层)

        # 初始化通道自注意力模块和自回归自注意力模块
        self.channel_attn_blocks=nn.ModuleList([ 
            channel_attn_block(enc_in,d_model,dropout)
            for _ in range(self.n_layers)
        ]) # 创建n_layers个channel_attn_block层，用于处理通道注意力，捕捉序列间的依赖关系。

        self.auto_attn_blocks=nn.ModuleList([
            auto_attn_block(enc_in,d_model,dropout)
            for _ in range(self.n_layers)
        ]) # 创建n_layers个auto_attn_block层，用于处理自回归注意力，捕捉序列内的变化。

        # 初始化位置编码层，对输入数据进行位置编码，使其符合输入要求。
        self.position_embedder=DataEmbedding(pe_type=pe_type,seq_len=seq_len, 
                                            d_model=d_model,c_in=enc_in) 
    
    def forward(self, inp):
        # 对输入数据进行位置编码，使其符合输入要求。
        # print(f'inp.shape_before_"inp=self.position_embedder(inp.permute(0,2,1)).permute(0,2,1)":{inp.shape}')
        inp=self.position_embedder(inp.permute(0,2,1)).permute(0,2,1) 
        # print(f'inp.shape_after_"inp=self.position_embedder(inp.permute(0,2,1)).permute(0,2,1)":{inp.shape}')
        main=self.LD(inp) 
        # main是卷积层的输出，即Trend分量，residual是原始输入减去Trend分量后得到的Seasonal分量。
        residual=inp-main
        # print(f'residual.shape_after_"residual=inp-main":{residual.shape}')
        res_1=residual # res_1和res_2分别通过自回归注意力和通道注意力，然后将两者相加作为最终输出。
        res_2=residual
        for i in range(self.n_layers):
            res_1=self.auto_attn_blocks[i](res_1)
        # print(f'res_1.shape_after_"res_1=self.auto_attn_blocks[i](res_1)":{res_1.shape}')
        for i in range(self.n_layers):
            res_2=self.channel_attn_blocks[i](res_2)
        # print(f'res_2.shape_after_"res_2=self.channel_attn_blocks[i](res_2)":{res_2.shape}')
        res=res_1+res_2 
        # print(f'res.shape_after_"res=res_1+res_2":{res.shape}')
        return res, main

class DyT(nn.Module):
    def __init__(self, C, init_alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(C))
        self.beta = nn.Parameter(torch.zeros(C))
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return self.gamma * x + self.beta

class channel_attn_block(nn.Module): # 通道注意力
    
    def __init__(self,enc_in,d_model,dropout):
        super(channel_attn_block, self).__init__()
        self.channel_att_norm=nn.BatchNorm1d(enc_in) # 通道注意力归一化。
        self.fft_norm=nn.LayerNorm(d_model) # 该层输出的归一化。
        # self.DyT_1 = DyT(enc_in)
        # self.DyT_2 = DyT(enc_in)
        # MultiheadAttention多头注意力机制，用于实现通道注意力机制。
        self.channel_attn=MultiheadAttention(d_model=d_model, n_heads=1,proj_dropout=dropout)
        self.fft_layer = nn.Sequential(
                                nn.Linear(d_model, int(d_model*2)),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(int(d_model*2), d_model),
                                ) # 全连接层？
        
    def forward(self, residual): 
        res_2=self.channel_att_norm(self.channel_attn(residual.permute(0,2,1))+residual.permute(0,2,1))
        res_2=self.fft_norm(self.fft_layer(res_2)+res_2) # 输入residual，首先经过通道注意力机制，再通过fft_layer。
        # res_2=self.DyT_1(self.channel_attn(residual.permute(0,2,1))+residual.permute(0,2,1))
        # res_2=self.DyT_2(self.fft_layer(res_2)+res_2) # 输入residual，首先经过通道注意力机制，再通过fft_layer。
        return res_2.permute(0,2,1)

class auto_attn_block(nn.Module): # 自回归注意力
    
    def __init__(self,enc_in,d_model,dropout):
        super(auto_attn_block, self).__init__()
        self.auto_attn_norm=nn.BatchNorm1d(enc_in)
        self.fft_norm=nn.LayerNorm(d_model)
        # 同上，不同之处在于注意力机制不同，这里用Auto_Attention初始化。
        # self.DyT_1 = DyT(enc_in)
        # self.DyT_2 = DyT(enc_in)
        self.auto_attn=Auto_Attention(P=64,d_model=d_model,proj_dropout=dropout)
        self.fft_layer = nn.Sequential(
                                nn.Linear(d_model, int(d_model*2)),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(int(d_model*2), d_model),
                                )
        
    def forward(self, residual):
        res_1=self.auto_attn_norm((self.auto_attn(residual)+residual).permute(0,2,1))
        res_1=self.fft_norm(self.fft_layer(res_1)+res_1)
        # res_1=self.DyT_1((self.auto_attn(residual)+residual).permute(0,2,1))
        # res_1=self.DyT_2(self.fft_layer(res_1)+res_1)
        return  res_1.permute(0,2,1)
    

class LD(nn.Module): # LD，Learnable Decomposition(卷积层)，用于实现文章提出的Leddam网络的第一步：将输入的时间序列数据分解为趋势和季节性两个部分
    
    def __init__(self,kernel_size=25): # 卷积核大小25，表示该卷积核在处理数据时会考虑当前点前后12个点的上下文信息。
        super(LD, self).__init__()
        
        # nn.Conv1d定义一个共享的一维卷积层，用于处理所有通道的数据，输入通道数和输出通道数均为1，步长为1。
        self.conv=nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=int(kernel_size//2), padding_mode='replicate', bias=True) 
        
        # 定义一种高斯分布型的卷积核权重
        kernel_size_half = kernel_size // 2
        sigma = 1.0  # sigma控制权重分布的平滑程度，1.0是默认值。
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size): # 卷积核权重被初始化为高斯分布，中心位置的权重最大，越往边缘权重越小。这种设计可以使卷积核在特征提取时对中心位置更敏感。
            weights[0, 0, i] = math.exp(-((i - kernel_size_half) / (2 * sigma)) ** 2)

        # 通过Softmax将卷积核权重转换为概率分布，以增强卷积核的学习能力？
        self.conv.weight.data = F.softmax(weights,dim=-1)
        self.conv.bias.data.fill_(0.0)
        
    def forward(self, inp):
        # 将输入数据的形状从(batch, N, T)调整为(batch, T, N)，适配一维卷积的输入要求。
        inp = inp.permute(0, 2, 1)
        
        # 将每个通道数据独立分离，形成多个(batch, T, 1)的矩阵。
        input_channels = torch.split(inp, 1, dim=1)
        
        # 对每个通道单独应用卷积操作self.conv(input_channel)，提取每个通道的趋势信息。
        conv_outputs = [self.conv(input_channel) for input_channel in input_channels]
        
        # 将各通道的卷积结果拼接为一个完整的矩阵，得到趋势输出矩阵。
        out = torch.cat(conv_outputs, dim=1)
        out = out.permute(0, 2, 1)
        return out
    
class Auto_Attention(nn.Module): 
    def __init__(self, P,d_model,proj_dropout=0.2): # 用于注意力初始化，定义QKV投影层
        """
        Initialize the Auto-Attention module.

        Args:
            d_model (int): The input and output dimension for queries, keys, and values.
        """
        super(Auto_Attention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_projector = nn.Sequential(nn.Linear(d_model, d_model),nn.Dropout(proj_dropout))
        self.P=P
        self.scale = nn.Parameter(torch.tensor(d_model ** -0.5), requires_grad=False)

    def auto_attention(self, inp): # 输入经过注意力机制，B批量大小，N特征数量，T序列长度，返回注意力输出
        """
        Perform auto-attention mechanism on the input.

        Args:
            inp (torch.Tensor): Input data of shape [B, N, T], where B is the batch size,
                               N is the number of features, and T is the sequence length.
        Returns:
            output (torch.Tensor): Output after auto-attention.
        """
        # Separate query and key
        query = self.W_Q(inp[:, :, 0, :].unsqueeze(-2))  # Query
        keys = self.W_K(inp)  # Keys
        values = self.W_V(inp)  # Values

        # Calculate dot product
        attn_scores = torch.matmul(query, keys.transpose(-2, -1)) * (self.scale)

        # Normalize attention scores
        attn_scores = F.softmax(attn_scores, dim=-1)
        assert torch.isnan(attn_scores).sum() == 0, print(attn_scores)

        # Weighted sum
        output = torch.matmul(attn_scores, values)

        return output

    def forward(self, inp): # 用于注意力机制的前向传播，P自回归行为的周期，返回自回归自注意力输出。
        """
        Forward pass of the Auto-Attention module.

        Args:
            P (int): The period for autoregressive behavior.
            inp (torch.Tensor): Input data of shape [B, T, N], where B is the batch size,
                               T is the sequence length, and N is the number of features.

        Returns:
            output (torch.Tensor): Output after autoregressive self-attention.
        """
        # Permute the input for further processing
        inp = inp.permute(0, 2, 1)  # [B, T, N] -> [B, N, T]

        T = inp.size(-1)

        cat_sequences = [inp]
        index = int(T / self.P) - 1 if T % self.P == 0 else int(T / self.P)

        for i in range(index):
            end = (i + 1) * self.P
            # Concatenate sequences to support autoregressive behavior
            cat_sequence = torch.cat([inp[:, :, end:], inp[:, :, 0:end]], dim=-1)
            cat_sequences.append(cat_sequence)

        # Stack the concatenated sequences
        output = torch.stack(cat_sequences, dim=-1)

        # Permute the output for attention calculation
        output = output.permute(0, 1, 3, 2)

        # Apply autoregressive self-attention
        output = self.auto_attention(output).squeeze(-2)
        output=self.out_projector(output).permute(0, 2, 1)
        
        return output
    
class MultiheadAttention(nn.Module): # 多头注意力
    def __init__(self, d_model, n_heads=1, attn_dropout=0., proj_dropout=0.2):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads 
        d_v = d_model // n_heads

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        # Scaled Dot-Product Attention (multiple heads)
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout)
        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                ):
        
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q
        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if prev is not None:
            output,prev = self.sdp_attn(q_s, k_s, v_s)
        else: output = self.sdp_attn(q_s, k_s, v_s)
        # output: [bs x n_heads x q_len x d_v]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)
        if prev is not None:
            return output,prev
        else: return output
   
class ScaledDotProductAttention(nn.Module): # 缩放点积注意力
    def __init__(self, d_model, n_heads, attn_dropout=0.):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=False)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''
        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * (self.scale)  # Scale

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None:
            attn_scores = attn_scores + prev
        # Normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights: [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # Compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]
        if prev is not None:
            return output,attn_scores
        else: return output


class DataEmbedding(nn.Module): # 用于数据嵌入，位置编码
    def __init__(self, pe_type,seq_len, d_model,c_in,dropout=0.):
        super(DataEmbedding, self).__init__()

        self.value_embedding = nn.Linear(seq_len, d_model)
        self.position_embedding = positional_encoding(pe=pe_type, learn_pe=True, q_len=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding
        return self.dropout(x)


# pos_encoding

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

def positional_encoding(pe, learn_pe, q_len, d_model): # pe返回值表示不同的位置编码方式。
    # Positional encoding
    if pe == None or pe == 'no':
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
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
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)

import torch
import torch.nn as nn
from layers.RevIN import RevIN
from layers.Leddam import Leddam

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.revin=configs.revin # 是否启用RevIN，标准化输入
        self.revin_layer=RevIN(channel=configs.enc_in,output_dim=configs.pred_len) # RevIN用于标准化和逆标准化，接收输入通道configs.enc_in和输出维度configs.pred_len
        self.leddam=Leddam(configs.enc_in,configs.seq_len,configs.d_model,
                       configs.dropout,configs.pe_type,kernel_size=25,n_layers=configs.n_layers)
        
        # 定义用于输出量级调整的线性层
        self.Linear_main = nn.Linear(configs.d_model, configs.pred_len) # 线性层，用于处理Leddam模块的Trend分量，生成左支路预测值。
        self.Linear_res = nn.Linear(configs.d_model, configs.pred_len) # 线性层，用于处理Leddam模块的Seasonal分量，生成右支路预测值。

        self.Linear_main.weight = nn.Parameter( # 设置self.Linear_main和self.Linear_res的权重为常数(1 / configs.d_model)。使用torch.ones初始化权重矩阵。
                (1 / configs.d_model) * torch.ones([configs.pred_len, configs.d_model])) 
        self.Linear_res.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([configs.pred_len, configs.d_model])) 
    
    def forward(self, inp): # 定义模型的前向传播过程，接受输入张量inp，返回预测值pred。
        # print(f'inp.shape_before_"if self.revin:":{inp.shape}')
        if self.revin:
            inp=self.revin_layer(inp)
        # print(f'inp.shape_after_"inp=self.revin_layer(inp)":{inp.shape}')
        res,main=self.leddam(inp) # 将标准化后的输入inp传入Leddam模块，得到main（Trend分量）和res（Seasonal分量）。
        # print(f'main.shepe_after_"res,main=self.leddam(inp)":{main.shape}')

        # 将趋势分量和季节性分量通过线性层映射到预测维度，得到对应的输出main_out和res_out。
        main_out=self.Linear_main(main.permute(0,2,1)).permute(0,2,1) 
        res_out=self.Linear_res(res.permute(0,2,1)).permute(0,2,1) # permute用于调整维度，以匹配线性层的输入格式。
        pred=main_out+res_out # 相加得最终预测值
        # print(f'pred.shepe_after_"pred=main_out+res_out":{pred.shape}')

        if self.revin: # 如果使用了RevIN，则将预测结果pred通过inverse_normalize方法逆标准化，还原为原始尺度。
            pred=self.revin_layer.inverse_normalize(pred)
            # print(f'pred.shepe_after_"pred=self.revin_layer.inverse_normalize(pred)":{pred.shape}')
        return pred


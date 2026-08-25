import torch
import torch.nn as nn

class RevIN(nn.Module): # 输入数据的归一化和逆归一化。
    
    def __init__(self,channel,output_dim): 
        super(RevIN, self).__init__()
        self.output_dim=output_dim # 指定输出维度

    def forward(self, x): # 标准化
        
        # Calculate mean and std along dim=1
        self.means = x.mean(1, keepdim=True).detach() # 沿dim=1维度计算x的均值，dim=1意为逐行计算，确保每个通道的均值都被计算。detach()？
        self.stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5) # 同上，计算x的标准差。
        
        # Normalize using learned parameters
        x_normalized = (x - self.means) / self.stdev # 使用计算出的均值和标准差对x进行归一化。
        return x_normalized
    
    def inverse_normalize(self, x_normalized): # 逆标准化
        x_normalized = x_normalized * \
                        (self.stdev[:, 0, :].unsqueeze(1).repeat(
                            1, self.output_dim, 1)) # 先缩放标准差。repeat()操作用于扩展标准差的维度，以匹配x_normalized的维度。
        x_normalized = x_normalized + \
                            (self.means[:, 0, :].unsqueeze(1).repeat(
                                1, self.output_dim, 1)) # 再将均值加回到x_normalized中，完成逆归一化。
        return x_normalized
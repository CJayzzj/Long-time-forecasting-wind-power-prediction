import os
import torch
from models import Leddam, LMTF_EIA, DLinear, iTransformer, PatchTST, Autoformer, FEDformer, TiDE, LMTF_EIA_ablation, LMTF_EIA_kernel, LMTF_EIA_DyT, LMTF_EIA_DyT_ablation, LMTF_EIA_DyT_kernel
from torchinfo import summary

class Exp_Basic(object): # 基础实验类，封装了模型的设备管理、构建和其他基本实验接口。作为实验类的基本框架，供具体实验类继承并实现具体的功能。
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Leddam': Leddam,
            'LMTF_EIA': LMTF_EIA,
            'DLinear': DLinear,
            'iTransformer': iTransformer,
            'PatchTST': PatchTST,
            'Autoformer': Autoformer,
            'FEDformer': FEDformer,
            'TiDE': TiDE,
            'LMTF_EIA_ablation': LMTF_EIA_ablation,  # LMTF-EIA消融实验模型
            'LMTF_EIA_kernel': LMTF_EIA_kernel,  # LMTF-EIA kernel_sizes超参数实验模型
            'LMTF_EIA_DyT': LMTF_EIA_DyT,  # LMTF-EIA 改进方向1: Dynamic Tanh 替代 LayerNorm
            'LMTF_EIA_DyT_ablation': LMTF_EIA_DyT_ablation,  # LMTF-EIA + DyT 消融实验模型
            'LMTF_EIA_DyT_kernel': LMTF_EIA_DyT_kernel,
        }
        self.device = self._acquire_device() # 确定设备（GPU 或 CPU）
        self.model = self._build_model().to(self.device) # 创建模型，并将其转移到指定的self.device上
        print(self.model)
        # torchinfo summary expects the forward signature to match the provided input shape
        if self.args.model in ['Leddam', 'LMTF_EIA', 'LMTF_EIA_ablation', 'LMTF_EIA_kernel', 'LMTF_EIA_DyT', 'LMTF_EIA_DyT_ablation', 'LMTF_EIA_DyT_kernel']:
            summary(model=self.model, input_size=(self.args.batch_size, self.args.seq_len, self.args.enc_in))  # 仅为Leddam和LMTF-EIA模型打印模型摘要
        else:
            print(f"Skip torchinfo summary for model {self.args.model} (needs 4 inputs: x_enc, x_mark_enc, x_dec, x_mark_dec)")        

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self): # 用于确定实验所使用的设备
        if self.args.use_gpu: # 检查self.args.use_gpu是否为True，以决定使用GPU还是CPU。
            os.environ["CUDA_VISIBLE_DEVICES"] = str( # 如果使用GPU，设置环境变量CUDA_VISIBLE_DEVICES来指定GPU设备。
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices # 如果不使用多GPU（self.args.use_multi_gpu为False），则仅使用self.args.gpu指定的GPU。否则，使用self.args.devices（指定多个GPU）以支持多GPU。
            device = torch.device('cuda:{}'.format(self.args.gpu)) # 设置device为torch.device('cuda:{})`，指向指定的GPU。
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else: 
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

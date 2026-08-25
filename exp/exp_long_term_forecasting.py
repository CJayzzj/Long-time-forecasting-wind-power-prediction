from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, metric_test
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def load_state_dict_compat(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        ckpt = ckpt['state_dict']
    if not isinstance(ckpt, dict):
        raise RuntimeError(f'Unsupported checkpoint format: {type(ckpt)}')

    keys = list(ckpt.keys())
    candidates = [ckpt]
    if keys and all(k.startswith('model.') for k in keys):
        candidates.append({k[len('model.'):]: v for k, v in ckpt.items()})
    else:
        candidates.append({f'model.{k}': v for k, v in ckpt.items()})

    errors = []
    for state_dict in candidates:
        try:
            model.load_state_dict(state_dict, strict=True)
            return
        except RuntimeError as e:
            errors.append(str(e))

    raise RuntimeError('Failed to load checkpoint with compatible key mapping.\n' + '\n---\n'.join(errors))

def visual(true, preds=None, name='./pic/test.pdf', dpi=600):
    """
    Results visualization
    """
    plt.figure()
    
    plt.plot(true, label='Ground Truth', linewidth=2, color='#ff7f0e')
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=1.8, color='#1f77b4')
    
    plt.legend()
    plt.savefig(name, bbox_inches='tight', dpi=dpi)

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()
        #model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'MSE' or self.args.loss == 'mse':
            criterion = nn.MSELoss()
        elif self.args.loss == 'MAE' or self.args.loss == 'mae':
            criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            preds=[]
            trues=[]
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device,non_blocking=True)
                batch_y = batch_y[:, -self.args.pred_len:,:].float()
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                if self.args.features == 'MS':
                    pred = outputs[:, :, -1].unsqueeze(-1).detach().cpu().numpy()
                    true = batch_y[:, :, -1].unsqueeze(-1).detach().numpy()
                elif self.args.features == 'MMS-151':
                    # 提取所有风机的P和OT (索引: 1,7,13,...,145,150)
                    p_indices = [1 + i*6 for i in range(25)]  # 25个风机的P索引
                    ot_index = 150  # OT的索引
                    selected_indices = p_indices + [ot_index]
                    pred = outputs[:, :, selected_indices].detach().cpu().numpy()
                    true = batch_y[:, :, selected_indices].detach().numpy()   
                elif self.args.features == 'MMS-26':
                    # 提取6个风机的ActivePower和OT
                    # 每个风机有4个变量，ActivePower是每组中的第3个变量
                    active_power_indices = [2 + i*4 for i in range(6)]  # 6个风机的ActivePower索引
                    ot_index = 25  # OT的索引（最后一个）
                    selected_indices = active_power_indices + [ot_index]
                    pred = outputs[:, :, selected_indices].detach().cpu().numpy()
                    true = batch_y[:, :, selected_indices].detach().numpy() 
                else: # M or S
                    pred = outputs.detach().cpu().numpy()
                    true = batch_y.detach().numpy()
                preds.append(pred)
                trues.append(true)
        if len(preds)>0:
            preds=np.concatenate(preds, axis=0)
            trues=np.concatenate(trues, axis=0)
        else:
            preds=preds[0]
            trues=trues[0]
        mse,mae= metric(preds, trues) 
        vali_loss=mae if criterion == 'MAE' or criterion == 'mae' else mse
        self.model.train()
        torch.cuda.empty_cache()
        return vali_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device,non_blocking=True)
                batch_y = batch_y[:, -self.args.pred_len:,:].float().to(self.device,non_blocking=True)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        if self.args.features == 'MS':
                            loss = criterion(outputs[:, :, -1].unsqueeze(-1), batch_y[:, :, -1].unsqueeze(-1))
                        elif self.args.features == 'MMS-151':
                            # 提取所有风机的P和OT
                            p_indices = [1 + i*6 for i in range(25)]
                            ot_index = 150
                            selected_indices = p_indices + [ot_index]
                            loss = criterion(outputs[:, :, selected_indices], batch_y[:, :, selected_indices])
                        elif self.args.features == 'MMS-26':
                            # 提取6个风机的ActivePower和OT
                            active_power_indices = [2 + i*4 for i in range(6)]
                            ot_index = 25
                            selected_indices = active_power_indices + [ot_index]
                            loss = criterion(outputs[:, :, selected_indices], batch_y[:, :, selected_indices])
                        else: # M or S             
                            loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x)
                    if self.args.features == 'MS':
                        loss = criterion(outputs[:, :, -1].unsqueeze(-1), batch_y[:, :, -1].unsqueeze(-1))
                    elif self.args.features == 'MMS-151':
                        # 提取所有风机的P和OT
                        p_indices = [1 + i*6 for i in range(25)]
                        ot_index = 150
                        selected_indices = p_indices + [ot_index]
                        loss = criterion(outputs[:, :, selected_indices], batch_y[:, :, selected_indices])
                    elif self.args.features == 'MMS-26':
                        # 提取6个风机的ActivePower和OT
                        active_power_indices = [2 + i*4 for i in range(6)]
                        ot_index = 25
                        selected_indices = active_power_indices + [ot_index]
                        loss = criterion(outputs[:, :, selected_indices], batch_y[:, :, selected_indices])
                    else: # M or S             
                        loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                torch.cuda.empty_cache()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss= self.vali(vali_data, vali_loader, self.args.loss)
            test_loss = self.vali(test_data, test_loader, self.args.loss)

            print("Epoch: {}, Steps: {} | Train Loss: {:.7f}  vali_loss: {:.7f}   test_loss: {:.7f} ".format(epoch + 1, train_steps, train_loss,  vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        torch.cuda.empty_cache()

    def test(self, setting, test=1):
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if test:
            print('loading model')
            load_state_dict_compat(self.model, os.path.join(path, 'checkpoint.pth'))

        self.model.eval()
        with torch.no_grad():
            preds=[]
            trues=[]
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device,non_blocking=True)
                batch_y = batch_y[:, -self.args.pred_len:,:].float()
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                if self.args.features == 'MS':
                    outputs = outputs[:, :, -1].unsqueeze(-1).detach().cpu().numpy()
                    batch_y = batch_y[:, :, -1].unsqueeze(-1).detach().numpy()
                elif self.args.features == 'MMS-151':
                    # 提取所有风机的P和OT
                    p_indices = [1 + i*6 for i in range(25)]
                    ot_index = 150
                    selected_indices = p_indices + [ot_index]
                    outputs = outputs[:, :, selected_indices].detach().cpu().numpy()
                    batch_y = batch_y[:, :, selected_indices].detach().numpy()
                elif self.args.features == 'MMS-26':
                    # 提取6个风机的ActivePower和OT
                    active_power_indices = [2 + i*4 for i in range(6)]
                    ot_index = 25
                    selected_indices = active_power_indices + [ot_index]
                    outputs = outputs[:, :, selected_indices].detach().cpu().numpy()
                    batch_y = batch_y[:, :, selected_indices].detach().numpy()
                else: # M or S
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().numpy()

                pred = outputs
                true = batch_y
                
                preds.append(pred)
                trues.append(true)
        if len(preds)>0:
            preds=np.concatenate(preds, axis=0)
            trues=np.concatenate(trues, axis=0)

        else:
            preds=preds[0]
            trues=trues[0]
        print('test shape:', preds.shape, trues.shape)

        visual_step = 720

        visual_path=f'./visual/{self.args.data_path[:-4]}/'
        os.makedirs(visual_path, exist_ok=True)
        output_file = os.path.join(visual_path, f'{self.args.model}_{self.args.data_path[:-4]}_sl{self.args.seq_len}_pl{self.args.pred_len}_n_layers_{self.args.n_layers}_d_model_{self.args.d_model}_dropout_{self.args.dropout}_pe_type_{self.args.pe_type}_bs_{self.args.batch_size}_lr_{self.args.learning_rate}.png')

        visual_true = np.zeros(self.args.seq_len + visual_step)
        for i in range(self.args.seq_len + visual_step):
            visual_true[i] = trues[i, 0, 0]

        visual_pred = np.zeros(visual_step)
        visual_pred = np.concatenate([np.full(self.args.seq_len, np.nan), visual_pred])
        for i in range (0, visual_step, self.args.pred_len):
            for j in range (self.args.pred_len):
                if i + self.args.seq_len + j < len(visual_pred):
                    visual_pred[i + self.args.seq_len + j] = preds[i + self.args.seq_len + 1, j, 0]
                else: 
                    break     

        visual(visual_true, visual_pred, name=output_file)

        # 新增：导出预测值和真实值（供后续可视化使用）
        npy_path = f'./test_dict/{self.args.data_path[:-4]}/'
        os.makedirs(npy_path, exist_ok=True)
        k_suffix = ''
        if self.args.model in ('LMTF_EIA_kernel', 'LMTF_EIA_DyT_kernel'):
            kernel_sizes = getattr(self.args, 'kernel_sizes', None)
            if kernel_sizes is None or kernel_sizes == 'default':
                k_val = 5
            else:
                k_val = len(str(kernel_sizes).split('_'))
            k_suffix = f'_K{k_val}'
        npy_prefix = (
            f'{self.args.model}'
            f'{k_suffix}'
            f'_sl{self.args.seq_len}'
            f'_pl{self.args.pred_len}'
        )
        np.save(os.path.join(npy_path, f'{npy_prefix}_pred.npy'), preds)
        np.save(os.path.join(npy_path, f'{npy_prefix}_true.npy'), trues)
        print(f'Saved pred/true npy to {npy_path}')

        dict_path= f'./test_dict/{self.args.data_path[:-4]}/'
        os.makedirs(dict_path, exist_ok=True)

        if self.args.features == 'MMS-151' or self.args.features == 'MMS-26':
            mse, mae, rmse, r2_mean, r2 = metric_test(preds, trues)
            print('mse:  {:.7f}  mae:  {:.7f}'.format(mse, mae))
            results_dict = {
                'md': self.args.model,
                'sl': self.args.seq_len,
                'pl': self.args.pred_len,
                'nl': self.args.n_layers,
                'dm': self.args.d_model,
                'dp': self.args.dropout,
                'bs': self.args.batch_size,
                'lr': self.args.learning_rate,
                'mse': "{:.7f}".format(mse),
                'mae': "{:.7f}".format(mae),
                'rmse': "{:.7f}".format(rmse),
                'r2_mean': "{:.7f}".format(r2_mean),
            }
            with open(os.path.join(dict_path, 'records.txt'), 'a') as f:
                result_line = "Results: "
                result_line += ", ".join([f"{key}: {value}" for key, value in results_dict.items()])
                print(result_line, file=f)
                # 换行后写入各个变量的R²
                r2_line = "R2_details: "
                r2_line += ", ".join([f"var_{i}: {r2_:.7f}" for i, r2_ in enumerate(r2)])
                print(r2_line, file=f)

        else:
            mse, mae = metric(preds, trues)
            print('mse:  {:.7f}  mae:  {:.7f}'.format(mse, mae))
            
            # 构建模型名称：如果是LMTF_EIA_kernel且指定了kernel_sizes，则添加配置信息
            model_name = self.args.model
            kernel_scale_str = None
            ablation_type_str = None
            
            if self.args.model == 'LMTF_EIA_kernel' and hasattr(self.args, 'kernel_sizes') and self.args.kernel_sizes != 'default':
                # 将kernel_sizes转换为更易读的格式，如 "25_13_7" -> "num3"
                kernel_list = self.args.kernel_sizes.split('_')
                num_kernels = len(kernel_list)
                model_name = f"LMTF-EIA_kernel_num{num_kernels}"
                # 生成kernel_scale格式: [49, 37, 25, 13, 7]
                kernel_scale_str = "[" + ", ".join(kernel_list) + "]"
            elif self.args.model == 'LMTF_EIA_ablation' and hasattr(self.args, 'ablation_type'):
                # 消融实验：使用论文名称 LMTF-EIA_{ablation_type}
                abl_type = self.args.ablation_type
                if abl_type == 'full':
                    model_name = "LMTF-EIA"
                else:
                    model_name = f"LMTF-EIA_{abl_type}"
                ablation_type_str = abl_type
            elif self.args.model == 'LMTF_EIA_DyT_ablation' and hasattr(self.args, 'ablation_type'):
                # DyT消融实验：使用论文名称 LMTF-EIA-DyT_{ablation_type}
                abl_type = self.args.ablation_type
                if abl_type == 'full':
                    model_name = "LMTF-EIA-DyT"
                else:
                    model_name = f"LMTF-EIA-DyT_{abl_type}"
                ablation_type_str = abl_type
            
            results_dict = {
                'md': model_name,
                'sl': self.args.seq_len,
                'pl': self.args.pred_len,
                'nl': self.args.n_layers,
                'dm': self.args.d_model,
                'dp': self.args.dropout,
                'bs': self.args.batch_size,
                'lr': self.args.learning_rate,
                'mse': "{:.7f}".format(mse),
                'mae': "{:.7f}".format(mae),
            }
            
            # 如果是LMTF_EIA_kernel，额外记录kernel_scale信息（更易读的格式）
            if kernel_scale_str is not None:
                results_dict['kernel_scale'] = kernel_scale_str
            
            with open(os.path.join(dict_path, 'records.txt'), 'a') as f:
                result_line = "Results: "
                result_line += ", ".join([f"{key}: {value}" for key, value in results_dict.items()])
                print(result_line, file=f)

        torch.cuda.empty_cache()
        return

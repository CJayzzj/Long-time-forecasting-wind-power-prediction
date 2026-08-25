import numpy as np
from sklearn.metrics import r2_score

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def R2(pred, true):
    # 确保pred和true是numpy数组
    pred = np.array(pred)
    true = np.array(true)
    
    # 获取形状信息
    batch_size, pred_len, num_variables = pred.shape
    
    # 计算每个变量的R²
    r2_scores = []
    for i in range(num_variables):
        # 将当前变量的所有预测值和真实值展平
        pred_flat = pred[:, :, i].flatten()
        true_flat = true[:, :, i].flatten()
        
        # 过滤掉NaN值
        mask = ~(np.isnan(true_flat) | np.isnan(pred_flat))
        true_filtered = true_flat[mask]
        pred_filtered = pred_flat[mask]
        
        # 如果过滤后没有数据，R²设为0
        if len(true_filtered) == 0:
            r2_scores.append(0.0)
            continue
            
        # 计算当前变量的R²
        r2 = r2_score(true_filtered, pred_filtered)
        r2_scores.append(r2)
    
    # 计算平均R²
    r2_mean = np.mean(r2_scores)

    return r2_mean, r2_scores

# def metric(pred, true):
#     mae = MAE(pred, true)
#     mse = MSE(pred, true)
#     rmse = RMSE(pred, true)
#     mape = MAPE(pred, true)
#     mspe = MSPE(pred, true)

#     return mae, mse, rmse, mape, mspe

def metric(pred, true):
    mse = MSE(pred, true)
    mae = MAE(pred, true)
    # rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    # r2_mean, r2 = R2(true, pred)

    return  mse, mae

def metric_test(pred, true):
    mse = MSE(pred, true)
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    r2_mean, r2 = R2(true, pred)

    return  mse, mae, rmse, r2_mean, r2

def metric_test_MMS(pred, true):
    """
    用于多变量多序列(MMS)特征类型，如 MMS-151 或 MMS-26
    返回: mse, mae, rmse, r2_mean, r2_list
    """
    mse = MSE(pred, true)
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    r2_mean, r2 = R2(true, pred)

    return mse, mae, rmse, r2_mean, r2

def metric_test_MS(pred, true):
    """
    用于单变量序列(MS)特征类型
    返回: mse, mae, rmse, r2
    """
    mse = MSE(pred, true)
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    r2_mean, _ = R2(true, pred)

    return mse, mae, rmse, r2_mean


from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_Special, Dataset_Special_T1, Dataset_Special_WTPGF
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'custom': Dataset_Custom,
    'Special': Dataset_Special,
    'Special_T1': Dataset_Special_T1,
    'Special_WTPGF': Dataset_Special_WTPGF,
}
def data_provider(args, flag):
    Data = data_dict[args.data] # 从data_dict中获取指定的数据集类，args.data是数据集名称，如'ETTh1'。

    if flag == 'test': 
        shuffle_flag = False # 如果是测试集，不打乱顺序
        drop_last = False # 保留最后一个批次，因为测试集中数据的顺序要保持一致。
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True # 丢弃最后一个批次，避免批次大小不一致
        batch_size = args.batch_size

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
    )
    print(flag, len(data_set)) # 输出数据集flag和数据集的样本数量？
    data_loader = DataLoader(
        data_set, # 传入的自定义数据集实例
        pin_memory=True, # 加速数据加载到GPU的传输
        batch_size=batch_size, # 每个批次的样本数
        shuffle=shuffle_flag, # 是否打乱数据
        num_workers=args.num_workers, # 数据加载的进程数量
        drop_last=drop_last) # 是否丢弃最后一个批次
    return data_set, data_loader

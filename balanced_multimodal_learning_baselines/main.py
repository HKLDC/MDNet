
import argparse
import os
import random
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch

from run import Run


MODEL = 'ogmge'
DATASET = 'fakesv'
SPLIT = 'temporal'


FOLD = 1


EPOCHES = 30
BATCH_SIZE = 16
NUM_WORKERS = 0
EPOCH_STOP = 5
SEED = 2025
GPU = 0
LR = 1e-4
DROPOUT = 0.1
WEIGHT_DECAY = 5e-5
EARLY_STOP = False
PATH_PARAM = './check_points/'
PATH_TENSORBOARD = './tb/'

# --- G-Blend ---
GBLEND_MODE = 'online'
GBLEND_SUPER_EPOCH = 5        # online 模式下每隔多少轮重新估计权重 (论文: 5)
GBLEND_WARMUP_EPOCHS = 10     # 预热轮数, 期间用均匀权重 (论文: 10)
GBLEND_EPS = 1e-3             # O² 的下限, 防止除零
GBLEND_METRIC = 'loss'        # 'loss' | 'acc' (用损失还是准确率估计 G/O)

# --- OGM-GE ---
OGM_ALPHA = 1.0               # 调制强度 α∈[0,1], 论文在验证集上选取
OGM_GE_SCALE = 1.0            # 泛化增强高斯噪声幅度 (0 关闭 GE, 退化为纯 OGM)
OGM_USE_UNI_LOSS = False      # 是否叠加单模态 CE 损失 (论文默认仅联合损失)
OGM_LAMBDA_UNI = 1.0          # 单模态损失的权重 (OGM_USE_UNI_LOSS=True 时生效)

# --- InfoReg ---
INFOREG_BETA = 0.9            # 调节强度 β (论文: 0.9 最优)
INFOREG_K = 0.04              # 关键学习窗口阈值 K (论文: 0.04 最优)
INFOREG_USE_UNI_LOSS = False  # 是否叠加单模态 CE 损失 (论文 InfoReg* 变体)
INFOREG_LAMBDA_UNI = 1.0      # 单模态损失的权重 (INFOREG_USE_UNI_LOSS=True 时生效)

# --- MLA ---
MLA_ALPHA = 1.0               # RLS 梯度修正矩阵 P 的正则项 α (论文 Eq. 5, 防止分母为零)


config = {
    'model': MODEL,
    'dataset': DATASET,
    'split': SPLIT,
    'fold': FOLD,
    'epoches': EPOCHES,
    'batch_size': BATCH_SIZE,
    'num_workers': NUM_WORKERS,
    'epoch_stop': EPOCH_STOP,
    'seed': SEED,
    'device': 'cuda' if GPU >= 0 else 'cpu',
    'gpu_index': GPU,
    'lr': LR,
    'dropout': DROPOUT,
    'weight_decay': WEIGHT_DECAY,
    'early_stop': EARLY_STOP,
    'path_param': PATH_PARAM,
    'path_tensorboard': PATH_TENSORBOARD,
    'method_cfg': {},
}


def build_method_cfg(model):
    if model == 'gblend':
        return {
            'gblend_mode': GBLEND_MODE,
            'gblend_super_epoch': GBLEND_SUPER_EPOCH,
            'gblend_warmup_epochs': GBLEND_WARMUP_EPOCHS,
            'gblend_eps': GBLEND_EPS,
            'gblend_metric': GBLEND_METRIC,
        }
    elif model == 'inforeg':
        return {
            'inforeg_beta': INFOREG_BETA,
            'inforeg_K': INFOREG_K,
            'inforeg_use_uni_loss': INFOREG_USE_UNI_LOSS,
            'inforeg_lambda_uni': INFOREG_LAMBDA_UNI,
        }
    elif model == 'ogmge':
        return {
            'ogm_alpha': OGM_ALPHA,
            'ogm_ge_scale': OGM_GE_SCALE,
            'ogm_use_uni_loss': OGM_USE_UNI_LOSS,
            'ogm_lambda_uni': OGM_LAMBDA_UNI,
        }
    elif model == 'mla':
        return {
            'mla_alpha': MLA_ALPHA,
        }
    else:
        raise ValueError("未知模型: {} (可选 'gblend' / 'inforeg' / 'ogmge' / 'mla')".format(model))


def parse_args(config):
    parser = argparse.ArgumentParser(description='短视频假新闻检测 - 模态不平衡基线')
    parser.add_argument('--model', default=config['model'], choices=['gblend', 'inforeg', 'ogmge', 'mla'])
    parser.add_argument('--dataset', default=config['dataset'], choices=['fakesv', 'fakett'])
    parser.add_argument('--split', default=config['split'], choices=['temporal', '5fold'])
    parser.add_argument('--fold', type=int, default=config['fold'])
    parser.add_argument('--epoches', type=int, default=config['epoches'])
    parser.add_argument('--batch_size', type=int, default=config['batch_size'])
    parser.add_argument('--seed', type=int, default=config['seed'])
    parser.add_argument('--gpu', type=int, default=GPU)
    parser.add_argument('--lr', type=float, default=config['lr'])
    args = parser.parse_args()
    config['model'] = args.model
    config['dataset'] = args.dataset
    config['split'] = args.split
    config['fold'] = args.fold
    config['epoches'] = args.epoches
    config['batch_size'] = args.batch_size
    config['seed'] = args.seed
    config['gpu_index'] = args.gpu
    config['device'] = 'cuda' if args.gpu >= 0 else 'cpu'
    config['lr'] = args.lr
    return config


if __name__ == '__main__':
    config = parse_args(config)

    config['method_cfg'] = build_method_cfg(config['model'])

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.get('gpu_index', GPU))

    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(config)
    Run(config).main()

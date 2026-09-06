from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.MultimodalBackbone import MultimodalBackbone
from model.MLAModel import MLAModel
from utils.dataloader import BSVFNDDataset, get_split_filenames, fakesv_collate_fn, fakett_collate_fn
from utils.tools import pretrain_bert_wwm_token, pretrain_bert_uncased_token
from utils.trainers import build_trainer
import numpy as np


def _init_fn(worker_id):
    np.random.seed(2024)


class Run():
    def __init__(self, config):
        self.model_param = config['model']          # 'gblend' | 'inforeg' | 'ogmge' | 'mla'
        self.dataset = config['dataset']            # 'fakesv' | 'fakett'
        self.split = config['split']                # 'temporal' | '5fold'
        self.fold = config['fold']
        self.epoches = config['epoches']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.epoch_stop = config['epoch_stop']
        self.seed = config['seed']
        self.device = config['device']
        self.lr = config['lr']
        self.save_param_dir = config['path_param']
        self.path_tensorboard = config['path_tensorboard']
        self.dropout = config['dropout']
        self.weight_decay = config['weight_decay']
        self.early_stop = config.get('early_stop', False)
        self.method_cfg = config.get('method_cfg', {})

    def get_dataloader(self):
        if self.dataset == 'fakesv':
            token = pretrain_bert_wwm_token()
            collate_fn = fakesv_collate_fn
        else:
            token = pretrain_bert_uncased_token()
            collate_fn = fakett_collate_fn

        files = get_split_filenames(self.split, self.fold)
        dataset_train = BSVFNDDataset(files['train'], token, self.dataset, split=self.split)
        dataset_val = BSVFNDDataset(files['val'], token, self.dataset, split=self.split)
        dataset_test = BSVFNDDataset(files['test'], token, self.dataset, split=self.split)

        train_dataloader = DataLoader(dataset_train, batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      pin_memory=True,
                                      shuffle=True,
                                      worker_init_fn=_init_fn,
                                      collate_fn=collate_fn)
        val_dataloader = DataLoader(dataset_val, batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    pin_memory=True,
                                    shuffle=False,
                                    worker_init_fn=_init_fn,
                                    collate_fn=collate_fn)
        test_dataloader = DataLoader(dataset_test, batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     pin_memory=True,
                                     shuffle=False,
                                     worker_init_fn=_init_fn,
                                     collate_fn=collate_fn)

        return dict(zip(['train', 'val', 'test'], [train_dataloader, val_dataloader, test_dataloader]))

    def get_model(self):
        if self.model_param == 'mla':
            return MLAModel(dataset=self.dataset, module_deep=2, dropout=self.dropout)
        return MultimodalBackbone(dataset=self.dataset, module_deep=2, dropout=self.dropout)

    def main(self):
        self.model = self.get_model()
        dataloaders = self.get_dataloader()
        trainer = build_trainer(
            self.model_param,
            model=self.model,
            device=self.device,
            lr=self.lr,
            dropout=self.dropout,
            dataloaders=dataloaders,
            weight_decay=self.weight_decay,
            epoch_stop=self.epoch_stop,
            epoches=self.epoches,
            save_param_path=self.save_param_dir + self.dataset + "/" + self.model_param + "/",
            writer=SummaryWriter(self.path_tensorboard),
            early_stop=self.early_stop,
            **self.method_cfg,
        )
        result = trainer.train()
        return result

import os
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from utils.metrics import metrics, get_confusionmatrix_fnd


class BaseTrainer:

    MODALITY_NAMES = ['text', 'visual', 'audio']

    def __init__(self, model, device, lr, dropout, dataloaders, weight_decay, save_param_path,
                 writer, epoch_stop, epoches, save_threshold=0.0, start_epoch=0, early_stop=False):
        self.model = model
        self.device = device
        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.epoch_stop = epoch_stop
        self.save_threshold = save_threshold
        self.writer = writer
        self.early_stop = early_stop

        if not os.path.exists(save_param_path):
            os.makedirs(save_param_path)
        self.save_param_path = save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None

    def forward_batch(self, epoch, phase, batch):
        raise NotImplementedError

    def after_backward(self, epoch, outputs):
        pass

    def on_epoch_start(self, epoch):
        pass

    def on_phase_end(self, phase, epoch, epoch_loss, results):
        pass


    def train(self):
        since = time.time()
        self.model = self.model.to(self.device)
        best_acc_test = 0.0
        best_epoch_test = 0
        is_earlystop = False
        last_save_path = ''

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch + 1, self.start_epoch + self.num_epochs))
            print('-' * 50)

            self.on_epoch_start(epoch)

            p = float(epoch) / 100
            lr = self.lr / (1. + 10 * p) ** 0.75
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)

            for phase in ['train', 'val', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                print('-' * 10)
                print(phase.upper())
                print('-' * 10)

                running_loss = 0.0
                tpred = []
                tlabel = []
                self._aux_preds = {}
                self._aux_labels = []

                for batch in tqdm(self.dataloaders[phase]):
                    batch_data = {k: v.to(self.device) for k, v in batch.items()}
                    label = batch_data['label']

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.forward_batch(epoch, phase, batch_data)
                        total_loss = outputs['loss']
                        logits = outputs['logits']

                        _, preds = torch.max(logits, 1)

                        if phase == 'train':
                            self.optimizer.zero_grad()
                            total_loss.backward()
                            self.after_backward(epoch, outputs)
                            self.optimizer.step()

                    tlabel.extend(label.detach().cpu().numpy().tolist())
                    tpred.extend(preds.detach().cpu().numpy().tolist())
                    running_loss += total_loss.item() * label.size(0)

                    # 累积辅助预测 (与主预测共享同一批标签)
                    aux = outputs.get('aux_preds')
                    if aux:
                        self._aux_labels.extend(label.detach().cpu().numpy().tolist())
                        for name, aux_preds in aux.items():
                            self._aux_preds.setdefault(name, []).extend(
                                aux_preds.detach().cpu().numpy().tolist())

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                print('Loss: {:.4f} '.format(epoch_loss))
                final_results = metrics(tlabel, tpred)
                print(final_results)
                get_confusionmatrix_fnd(tpred, tlabel)
                for name, aux_preds in self._aux_preds.items():
                    aux_results = metrics(self._aux_labels, aux_preds)
                    print('  [{}] {}'.format(name, aux_results))
                    get_confusionmatrix_fnd(aux_preds, self._aux_labels)
                self._aux_preds = {}
                self._aux_labels = []

                if phase == 'test' and final_results['acc'] > best_acc_test:
                    best_acc_test = final_results['acc']
                    best_epoch_test = epoch + 1
                    if best_acc_test > self.save_threshold:
                        if os.path.exists(last_save_path):
                            print('delete the previous checkpoint...')
                            os.remove(last_save_path)
                        save_path = self.save_param_path + "_test_epoch" + str(best_epoch_test) + "_{0:.4f}".format(best_acc_test)
                        torch.save(self.model.state_dict(), save_path)
                        last_save_path = save_path
                        print("saved " + save_path)
                    elif self.early_stop and epoch - best_epoch_test >= self.epoch_stop - 1:
                        is_earlystop = True
                        print("early stopping...")

                self.on_phase_end(phase, epoch, epoch_loss, final_results)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print("Best model on test: epoch" + str(best_epoch_test) + "_" + str(best_acc_test))
        return True

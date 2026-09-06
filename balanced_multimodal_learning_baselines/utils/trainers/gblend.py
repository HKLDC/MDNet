
import numpy as np
import torch
from torch import nn

from utils.trainers.base import BaseTrainer


class GBlendTrainer(BaseTrainer):
    def __init__(self, model, device, lr, dropout, dataloaders, weight_decay, save_param_path,
                 writer, epoch_stop, epoches, save_threshold=0.0, start_epoch=0, early_stop=False,
                 gblend_mode='online', gblend_super_epoch=5, gblend_warmup_epochs=10,
                 gblend_eps=1e-3, gblend_metric='loss'):
        super().__init__(model, device, lr, dropout, dataloaders, weight_decay, save_param_path,
                         writer, epoch_stop, epoches, save_threshold=save_threshold,
                         start_epoch=start_epoch, early_stop=early_stop)
        assert gblend_mode in ('online', 'offline'), "gblend_mode 仅支持 'online' / 'offline'"
        assert gblend_metric in ('loss', 'acc'), "gblend_metric 仅支持 'loss' / 'acc'"

        self.heads = self.MODALITY_NAMES + ['joint']          # text / visual / audio / joint
        self.mode = gblend_mode
        self.super_epoch = gblend_super_epoch
        self.warmup_epochs = gblend_warmup_epochs
        self.eps = gblend_eps
        self.metric = gblend_metric

        # 每轮各头的训练/验证损失与准确率曲线 (用于 O/G 估计)
        self.head_losses = {'train': {h: [] for h in self.heads}, 'val': {h: [] for h in self.heads}}
        self.head_accs = {'train': {h: [] for h in self.heads}, 'val': {h: [] for h in self.heads}}

        # 当前生效的损失权重, 初始为均匀权重
        self.weights = {h: 1.0 / len(self.heads) for h in self.heads}
        self._estimated = False

        # phase 内累积器
        self._acc_loss = {'train': {h: 0.0 for h in self.heads}, 'val': {h: 0.0 for h in self.heads}}
        self._acc_correct = {'train': {h: 0 for h in self.heads}, 'val': {h: 0 for h in self.heads}}
        self._acc_count = {'train': 0, 'val': 0}

    def forward_batch(self, epoch, phase, batch):
        out = self.model(**batch)
        label = batch['label']

        head_losses = {}
        head_preds = {}
        for h in self.heads:
            logits = out['joint_logits'] if h == 'joint' else out['{}_logits'.format(h)]
            head_losses[h] = self.criterion(logits, label)
            head_preds[h] = torch.max(logits, 1)[1]

        total_loss = sum(self.weights[h] * head_losses[h] for h in self.heads)

        if phase in ('train', 'val'):
            n = label.size(0)
            for h in self.heads:
                self._acc_loss[phase][h] += head_losses[h].item() * n
                self._acc_correct[phase][h] += (head_preds[h] == label).sum().item()
            self._acc_count[phase] += n

        return {'loss': total_loss, 'logits': out['joint_logits']}

    def on_phase_end(self, phase, epoch, epoch_loss, results):
        if phase in ('train', 'val'):
            n = max(self._acc_count[phase], 1)
            for h in self.heads:
                self.head_losses[phase][h].append(self._acc_loss[phase][h] / n)
                self.head_accs[phase][h].append(self._acc_correct[phase][h] / n)
            self._acc_loss[phase] = {h: 0.0 for h in self.heads}
            self._acc_correct[phase] = {h: 0 for h in self.heads}
            self._acc_count[phase] = 0

        # 权重估计时机: 预热结束后
        if phase == 'val' and epoch >= self.warmup_epochs:
            if self.mode == 'offline':
                if not self._estimated:
                    self.estimate_weights()
                    self._estimated = True
            else:  # online
                if (epoch - self.warmup_epochs) % self.super_epoch == 0:
                    self.estimate_weights()

    def estimate_weights(self):
        w = {}
        for h in self.heads:
            if self.metric == 'loss':
                G = self.head_losses['val'][h][0] - self.head_losses['val'][h][-1]
                O = (self.head_losses['train'][h][0] - self.head_losses['train'][h][-1]) - G
            else:  # 'acc'
                G = self.head_accs['val'][h][-1] - self.head_accs['val'][h][0]
                O = (self.head_accs['train'][h][-1] - self.head_accs['train'][h][0]) - G
            w[h] = max(G, 0.0) / max(O * O, self.eps)

        total = sum(w.values())
        if total <= 0:
            self.weights = {h: 1.0 / len(self.heads) for h in self.heads}
            print('[G-Blend] 权重估计退化, 回退到均匀权重')
        else:
            self.weights = {h: w[h] / total for h in self.heads}
        print('[G-Blend] 损失权重 - ' + ', '.join('{}={:.3f}'.format(h, self.weights[h]) for h in self.heads))

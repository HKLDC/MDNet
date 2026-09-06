
import math

import torch

from utils.trainers.base import BaseTrainer


class InfoRegTrainer(BaseTrainer):
    def __init__(self, model, device, lr, dropout, dataloaders, weight_decay, save_param_path,
                 writer, epoch_stop, epoches, save_threshold=0.0, start_epoch=0, early_stop=False,
                 inforeg_beta=0.9, inforeg_K=0.04, inforeg_use_uni_loss=False, inforeg_lambda_uni=1.0):
        super().__init__(model, device, lr, dropout, dataloaders, weight_decay, save_param_path,
                         writer, epoch_stop, epoches, save_threshold=save_threshold,
                         start_epoch=start_epoch, early_stop=early_stop)
        self.beta = inforeg_beta          # 调节强度超参数 (论文: β=0.9 最优)
        self.K = inforeg_K                # 关键学习窗口阈值 (论文: K=0.04 最优)
        self.use_uni_loss = inforeg_use_uni_loss
        self.lambda_uni = inforeg_lambda_uni

        self.encoders = {
            'text': self.model.text_encoder,
            'visual': self.model.visual_encoder,
            'audio': self.model.audio_encoder,
        }

        # 每轮各模态的 Fisher 信息迹 Tr(F^t_m) (按 epoch 记录)
        self.fisher_trace = {m: [] for m in self.MODALITY_NAMES}
        # 当前轮 batch 内累积的梯度平方范数
        self._sq_grad_sum = {m: 0.0 for m in self.MODALITY_NAMES}
        self._train_batches = 0

        # 关键学习窗口标记 (每轮开始更新)
        self.prime_window = {m: False for m in self.MODALITY_NAMES}
        # 上一轮 (t-1) 各模态编码器参数 (detached), 用于调节项 ||w - w_prev||²
        self.prev_params = {m: None for m in self.MODALITY_NAMES}

    def on_epoch_start(self, epoch):
        if epoch >= 2:
            for m in self.MODALITY_NAMES:
                if len(self.fisher_trace[m]) >= 2:
                    F_prev1 = self.fisher_trace[m][-1]
                    F_prev2 = self.fisher_trace[m][-2]
                    rate = (F_prev1 - F_prev2) / max(F_prev1, 1e-8)
                    self.prime_window[m] = rate > self.K
                else:
                    self.prime_window[m] = False
        else:
            for m in self.MODALITY_NAMES:
                self.prime_window[m] = False

    def forward_batch(self, epoch, phase, batch):
        out = self.model(**batch)
        label = batch['label']

        total_loss = self.criterion(out['joint_logits'], label)
        if self.use_uni_loss:
            uni_loss = sum(self.criterion(out['{}_logits'.format(m)], label) for m in self.MODALITY_NAMES)
            total_loss = total_loss + self.lambda_uni * uni_loss

        # 性能分数 (Eq. 8): 批内 CE 均值, 越低表示该模态表现越好
        scores = {m: self.criterion(out['{}_logits'.format(m)], label).item()
                  for m in self.MODALITY_NAMES}
        delta = self._performance_gap(scores)

        if phase == 'train':
            reg = 0.0
            for m in self.MODALITY_NAMES:
                if self.prime_window[m] and delta[m] > 0 and self.prev_params[m] is not None:
                    alpha_m = math.exp(self.beta * math.tanh(delta[m]))
                    diff = 0.0
                    for p, p_prev in zip(self.encoders[m].parameters(), self.prev_params[m]):
                        diff = diff + ((p - p_prev) ** 2).sum()
                    reg = reg + (alpha_m / 2.0) * diff
            if reg > 0:
                total_loss = total_loss + reg

        return {'loss': total_loss, 'logits': out['joint_logits']}

    def after_backward(self, epoch, outputs):
        for m, enc in self.encoders.items():
            s = 0.0
            for p in enc.parameters():
                if p.grad is not None:
                    s += (p.grad ** 2).sum().item()
            self._sq_grad_sum[m] += s
        self._train_batches += 1

    def on_phase_end(self, phase, epoch, epoch_loss, results):
        if phase == 'train':
            n = max(self._train_batches, 1)
            for m in self.MODALITY_NAMES:
                self.fisher_trace[m].append(self._sq_grad_sum[m] / n)
                self._sq_grad_sum[m] = 0.0
            self._train_batches = 0
            # 保存本轮参数快照, 作为下一轮的 w^{t-1}
            self.prev_params = {m: [p.detach().clone() for p in self.encoders[m].parameters()]
                                for m in self.MODALITY_NAMES}
            trace_str = ', '.join('{}={:.4f}'.format(m, self.fisher_trace[m][-1])
                                  for m in self.MODALITY_NAMES)
            win_str = ', '.join('{}={}'.format(m, self.prime_window[m]) for m in self.MODALITY_NAMES)
            print('[InfoReg] Tr(F): ' + trace_str)
            print('[InfoReg] prime-window: ' + win_str)

    # ---------------------------------------------------------------
    def _performance_gap(self, scores):
        delta = {}
        for m in self.MODALITY_NAMES:
            worse = [scores[m2] for m2 in self.MODALITY_NAMES
                     if m2 != m and scores[m2] > scores[m]]
            delta[m] = (1.0 / len(worse)) * sum(s2 - scores[m] for s2 in worse) if worse else 0.0
        return delta

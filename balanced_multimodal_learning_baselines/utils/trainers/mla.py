
import torch
import torch.nn.functional as F

from utils.trainers.base import BaseTrainer


class MLATrainer(BaseTrainer):
    def __init__(self, model, device, lr, dropout, dataloaders, weight_decay, save_param_path,
                 writer, epoch_stop, epoches, save_threshold=0.0, start_epoch=0, early_stop=False,
                 mla_alpha=1.0):
        super().__init__(model, device, lr, dropout, dataloaders, weight_decay, save_param_path,
                         writer, epoch_stop, epoches, save_threshold=save_threshold,
                         start_epoch=start_epoch, early_stop=early_stop)
        self.alpha = mla_alpha

        self.fea_dim = self.model.fea_dim
        # 梯度修正矩阵 P (s×s), 初始化为单位阵 (Eq. 5, P_0 = I)
        self.P = torch.eye(self.fea_dim, device=self.device)
        self.iteration = 0  # 全局迭代计数 (跨 epoch 递增, 用于交替模态分配)

        self._cur_feat = None

    def forward_batch(self, epoch, phase, batch):
        if phase == 'train':
            m = self.model.MODALITY_NAMES[self.iteration % len(self.model.MODALITY_NAMES)]
            logits, fea = self.model.forward_modality(m, **batch)
            loss = self.criterion(logits, batch['label'])
            self._cur_feat = fea.detach()
            return {'loss': loss, 'logits': logits, 'modality': m}

        out = self.model(**batch)
        label = batch['label']
        fused_logits, per_logits = self._uncertainty_fusion(out)
        loss = self.criterion(fused_logits, label)
        aux_preds = {m: torch.max(per_logits[m], 1)[1] for m in self.model.MODALITY_NAMES}
        return {'loss': loss, 'logits': fused_logits, 'aux_preds': aux_preds}

    def after_backward(self, epoch, outputs):
        fea = self._cur_feat
        h_bar = fea.mean(dim=0)
        Ph = torch.mv(self.P, h_bar)
        denom = self.alpha + torch.dot(h_bar, Ph)
        q = Ph / denom
        self.P = self.P - torch.outer(q, Ph)

        w_grad = self.model.shared_head.weight.grad
        if w_grad is not None:
            with torch.no_grad():
                self.model.shared_head.weight.grad.data = torch.matmul(w_grad, self.P)

        self.iteration += 1

    def on_phase_end(self, phase, epoch, epoch_loss, results):
        if phase == 'train':
            m = self.model.MODALITY_NAMES[(self.iteration - 1) % len(self.model.MODALITY_NAMES)]
            print('[MLA] 当前迭代模态: {}, 梯度修正矩阵 P 对角线均值: {:.4f}'.format(
                m, torch.diag(self.P).mean().item()))

    def _uncertainty_fusion(self, out):
        logits = {m: out['{}_logits'.format(m)] for m in self.model.MODALITY_NAMES}
        entropies = []
        for m in self.model.MODALITY_NAMES:
            p = F.softmax(logits[m], dim=-1)
            entropies.append(-(p * torch.log(torch.clamp(p, min=1e-8))).sum(dim=-1))
        e = torch.stack(entropies, dim=1)                  # [B, M]
        e_max = e.max(dim=1, keepdim=True).values
        weights = F.softmax(e_max - e, dim=1)              # [B, M]
        logits_stack = torch.stack([logits[m] for m in self.model.MODALITY_NAMES], dim=1)  # [B, M, C]
        fused = (weights.unsqueeze(-1) * logits_stack).sum(dim=1)
        return fused, logits

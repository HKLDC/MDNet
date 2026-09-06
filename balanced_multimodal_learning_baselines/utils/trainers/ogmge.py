
import torch
import torch.nn.functional as F

from utils.trainers.base import BaseTrainer


class OGMGETrainer(BaseTrainer):
    def __init__(self, model, device, lr, dropout, dataloaders, weight_decay, save_param_path,
                 writer, epoch_stop, epoches, save_threshold=0.0, start_epoch=0, early_stop=False,
                 ogm_alpha=1.0, ogm_ge_scale=1.0, ogm_use_uni_loss=False, ogm_lambda_uni=1.0):
        super().__init__(model, device, lr, dropout, dataloaders, weight_decay, save_param_path,
                         writer, epoch_stop, epoches, save_threshold=save_threshold,
                         start_epoch=start_epoch, early_stop=early_stop)
        self.alpha = ogm_alpha            # 调制强度超参数 α∈[0,1] (论文在验证集上选取)
        self.ge_scale = ogm_ge_scale      # 高斯噪声幅度, 0 表示关闭 GE
        self.use_uni_loss = ogm_use_uni_loss
        self.lambda_uni = ogm_lambda_uni

        self.coeff = {m: 1.0 for m in self.MODALITY_NAMES}  # 当前 batch 各模态梯度系数 k_m
        self.encoders = {
            'text': self.model.text_encoder,
            'visual': self.model.visual_encoder,
            'audio': self.model.audio_encoder,
        }

    def forward_batch(self, epoch, phase, batch):
        out = self.model(**batch)
        label = batch['label']

        total_loss = self.criterion(out['joint_logits'], label)
        if self.use_uni_loss:
            uni_loss = sum(self.criterion(out['{}_logits'.format(m)], label) for m in self.MODALITY_NAMES)
            total_loss = total_loss + self.lambda_uni * uni_loss

        scores = {}
        for m in self.MODALITY_NAMES:
            prob = F.softmax(out['{}_logits'.format(m)], dim=-1)
            scores[m] = prob.gather(1, label.unsqueeze(1)).mean().item()

        eps = 1e-8
        for m in self.MODALITY_NAMES:
            others = [scores[m2] for m2 in self.MODALITY_NAMES if m2 != m]
            rho = scores[m] / (sum(others) / len(others) + eps)
            if rho > 1:
                self.coeff[m] = (1.0 - torch.tanh(torch.tensor(self.alpha * rho)).item())
            else:
                self.coeff[m] = 1.0

        return {'loss': total_loss, 'logits': out['joint_logits']}

    def after_backward(self, epoch, outputs):
        for m, enc in self.encoders.items():
            k = self.coeff[m]
            for p in enc.parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach()
                p.grad.data.mul_(k)
                if self.ge_scale > 0:
                    noise = torch.randn_like(g) * g
                    p.grad.data.add_(self.ge_scale * noise)

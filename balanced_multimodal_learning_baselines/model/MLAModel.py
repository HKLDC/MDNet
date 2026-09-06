import torch
from torch import nn

from utils.tools import pretrain_bert_wwm_model, pretrain_bert_uncased_model
from model.MultimodalBackbone import ModalityEncoder


class MLAModel(nn.Module):

    MODALITY_NAMES = ['text', 'visual', 'audio']

    def __init__(self, dataset, module_deep=2, dropout=0.1, fea_dim=128):
        super().__init__()
        self.dataset = dataset
        self.fea_dim = fea_dim

        if dataset == 'fakesv':
            self.bert = pretrain_bert_wwm_model()      # 冻结, 隐藏维度 1024
            text_dim = 1024
        else:
            self.bert = pretrain_bert_uncased_model()  # 冻结, 隐藏维度 768
            text_dim = 768

        self.text_encoder = ModalityEncoder('Text', in_dim=text_dim, fea_dim=fea_dim,
                                            module_deep=module_deep, dropout=dropout)
        self.visual_encoder = ModalityEncoder('Visual', in_dim=1024, fea_dim=fea_dim,
                                              module_deep=module_deep, dropout=dropout)
        self.audio_encoder = ModalityEncoder('Audio', in_dim=1024, fea_dim=fea_dim,
                                             module_deep=module_deep, dropout=dropout)

        # 跨模态共享头 g: 单层全连接 (论文: shared head 为 fully connected layer)
        self.shared_head = nn.Linear(fea_dim, 2)

    @property
    def encoders(self):
        return {
            'text': self.text_encoder,
            'visual': self.visual_encoder,
            'audio': self.audio_encoder,
        }

    def get_feature(self, modality, **kwargs):
        if modality == 'text':
            title_inputid = kwargs['title_inputid']
            title_mask = kwargs['title_mask']
            text_seq = self.bert(title_inputid, attention_mask=title_mask)['last_hidden_state']
            return self.text_encoder(text_seq)
        elif modality == 'visual':
            return self.visual_encoder(kwargs['frames'])
        elif modality == 'audio':
            return self.audio_encoder(kwargs['audio_feas'])
        else:
            raise ValueError("未知模态: {}".format(modality))

    def forward_modality(self, modality, **kwargs):
        fea = self.get_feature(modality, **kwargs)   # [B, fea_dim]
        logits = self.shared_head(fea)               # [B, 2]
        return logits, fea

    def forward(self, **kwargs):
        """推理: 返回所有模态各自通过共享头的 logits 与特征 (供不确定性融合)。"""
        out = {}
        for m in self.MODALITY_NAMES:
            logits, fea = self.forward_modality(m, **kwargs)
            out['{}_logits'.format(m)] = logits
            out['{}_feat'.format(m)] = fea
        return out

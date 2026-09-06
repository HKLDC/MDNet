import torch
from torch import nn
import torch.nn.functional as F

from utils.tools import pretrain_bert_wwm_model, pretrain_bert_uncased_model
from model.GL_Transformer import GL_Transformer


class AttnPooling(nn.Module):

    def __init__(self, feat_dim=512, compressed_dim=128):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, feat_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=4,
            kdim=feat_dim,
            vdim=feat_dim,
        )
        self.proj = nn.Linear(feat_dim, compressed_dim)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.query.expand(B, -1, -1)  # [B, 1, feat_dim]
        attn_output, _ = self.attn(
            query=q.transpose(0, 1),  # [1, B, feat_dim]
            key=x.transpose(0, 1),    # [L, B, feat_dim]
            value=x.transpose(0, 1),  # [L, B, feat_dim]
        )
        return self.proj(attn_output.squeeze(0))


class ModalityEncoder(nn.Module):

    def __init__(self, modality, in_dim, trans_dim=512, fea_dim=128, module_deep=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.modality = modality
        self.in_dim = in_dim
        self.linear = nn.Sequential(
            nn.Linear(in_dim, trans_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.learner = GL_Transformer(
            trans_dim, module_deep, num_heads, trans_dim // 2, trans_dim * 2, dropout=0.1
        )
        self.pool = AttnPooling(feat_dim=trans_dim, compressed_dim=fea_dim)

    def forward(self, x):
        h = self.linear(x)
        h = self.learner(h)
        return self.pool(h)


class MultimodalBackbone(nn.Module):
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

        # 晚期拼接融合分类器: 3 * fea_dim -> 2
        self.joint_classifier = nn.Linear(3 * fea_dim, 2)

    @property
    def modality_names(self):
        return ['text', 'visual', 'audio']

    def forward(self, **kwargs):
        # ---- 文本: 冻结 BERT 提取序列特征 ----
        title_inputid = kwargs['title_inputid']
        title_mask = kwargs['title_mask']
        text_seq = self.bert(title_inputid, attention_mask=title_mask)['last_hidden_state']
        f_t = self.text_encoder(text_seq)

        frames = kwargs['frames']
        f_v = self.visual_encoder(frames)

        audio_feas = kwargs['audio_feas']
        f_a = self.audio_encoder(audio_feas)
        cat = torch.cat([f_t, f_v, f_a], dim=-1)
        joint_logits = self.joint_classifier(cat)

        W = self.joint_classifier.weight
        b = self.joint_classifier.bias
        logits_t = F.linear(f_t, W[:, :self.fea_dim], b / 3.0)
        logits_v = F.linear(f_v, W[:, self.fea_dim:2 * self.fea_dim], b / 3.0)
        logits_a = F.linear(f_a, W[:, 2 * self.fea_dim:], b / 3.0)

        return {
            'joint_logits': joint_logits,
            'text_logits': logits_t,
            'visual_logits': logits_v,
            'audio_logits': logits_a,
            'text_feat': f_t,
            'visual_feat': f_v,
            'audio_feat': f_a,
        }

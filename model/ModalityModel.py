import torch
from torch import nn
from utils.tools import *
from model.GL_Transformer import GL_Transformer
from model.Transformer_conv import Transformer_conv, get_clones
from model.Transformer_ffn import Transformer_ffn

class AttnPooling(nn.Module):
    def __init__(self, feat_dim=512, compressed_dim=128):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, feat_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=4,
            kdim=feat_dim,
            vdim=feat_dim
        )
        # squeeze
        self.proj = nn.Linear(feat_dim, compressed_dim)

    def forward(self, x):
        B, L, _ = x.shape

        q = self.query.expand(B, -1, -1)  # [B, 1, 512]

        attn_output, _ = self.attn(
            query=q.transpose(0, 1),  # [1, B, 512]
            key=x.transpose(0, 1),    # [L, B, 512]
            value=x.transpose(0, 1)   # [L, B, 512]
        )
        
        # [1,B,512] → [B,512] → [B,128]
        return self.proj(attn_output.squeeze(0))
    
class ModalityModel(torch.nn.Module):
    def __init__(self,dataset,modality,module_deep,dropout):
        super(ModalityModel, self).__init__()
        if dataset == 'fakesv':
            self.bert = pretrain_bert_wwm_model()
            self.text_dim = 1024
        else:
            self.bert = pretrain_bert_uncased_model()
            self.text_dim = 768

        self.img_dim = 1024
        self.hubert_dim = 1024
        self.video_dim = 1024
        self.num_frames = 83
        self.num_audioframes = 50
        self.num_heads = 8
        self.trans_dim = 512
        self.fea_dim = 128
        self.dropout = dropout
        self.modality = modality
        self.module_deep = module_deep
        
        if self.modality == 'Text':
            self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, self.trans_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
            self.text_learner = GL_Transformer(self.trans_dim, self.module_deep, self.num_heads, self.trans_dim // 2, self.trans_dim * 2,
                                           dropout=0.1)
        elif self.modality == 'Visual':
            self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, self.trans_dim), torch.nn.ReLU(),
                                        nn.Dropout(p=self.dropout))
            self.visual_learner = GL_Transformer(self.trans_dim, self.module_deep, self.num_heads, self.trans_dim // 2, self.trans_dim * 2,
                                             dropout=0.1)
        elif self.modality == 'Audio':
            self.linear_audio = nn.Sequential(torch.nn.Linear(self.hubert_dim, self.trans_dim), torch.nn.ReLU(),
                                          nn.Dropout(p=self.dropout))
            self.audio_learner = GL_Transformer(self.trans_dim, self.module_deep, self.num_heads, self.trans_dim // 2, self.trans_dim * 2,
                                            dropout=0.1)
        else:
            self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, self.trans_dim), torch.nn.ReLU(),
                                             nn.Dropout(p=self.dropout))
            self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, self.trans_dim), torch.nn.ReLU(),
                                            nn.Dropout(p=self.dropout))
            self.linear_audio = nn.Sequential(torch.nn.Linear(self.hubert_dim, self.trans_dim), torch.nn.ReLU(),
                                              nn.Dropout(p=self.dropout))
            self.shared_learner = Transformer_conv(self.trans_dim, self.module_deep, self.num_heads, self.trans_dim * 2, dropout=0.1)
            self.mix_learner = Transformer_ffn(self.trans_dim, self.module_deep, self.num_heads, self.trans_dim * 2, dropout=0.1)
        
        self.pool_learner = AttnPooling(feat_dim=self.trans_dim, compressed_dim=self.fea_dim)
        self.classifier = nn.Linear(self.fea_dim,2)
    
    def forward(self,  **kwargs):
        if self.modality == 'Text':
            ### Title ###
            title_inputid = kwargs['title_inputid']  # (batch,512)
            title_mask = kwargs['title_mask']  # (batch,512)
            fea_text = self.bert(title_inputid, attention_mask=title_mask)['last_hidden_state']  # (batch,sequence,768)
            fea_text = self.linear_text(fea_text)

            fea_text = self.text_learner(fea_text)
            fea = self.pool_learner(fea_text)

        elif self.modality == 'Visual':
            ### Image Frames ###
            frames = kwargs['frames']  # (batch,30,4096)
            fea_img = self.linear_img(frames)

            fea_img = self.visual_learner(fea_img)
            fea = self.pool_learner(fea_img)

        elif self.modality == 'Audio':
            ### Audio Frames ###
            audioframes = kwargs['audio_feas']  # (batch,36,12288)
            fea_audio = self.linear_audio(audioframes)

            fea_audio = self.audio_learner(fea_audio)
            fea = self.pool_learner(fea_audio)
        else:
            ### Title ###
            title_inputid = kwargs['title_inputid']  # (batch,512)
            title_mask = kwargs['title_mask']  # (batch,512)
            fea_text = self.bert(title_inputid, attention_mask=title_mask)['last_hidden_state']  # (batch,sequence,768)
            fea_text = self.linear_text(fea_text)

            ### Audio Frames ###
            audioframes = kwargs['audio_feas']  # (batch,36,12288)
            fea_audio = self.linear_audio(audioframes)

            ### Image Frames ###
            frames = kwargs['frames']  # (batch,30,4096)
            fea_img = self.linear_img(frames)

            # 使用共享编码器学习共享表示
            fea_text_c = self.shared_learner(fea_text)
            fea_img_c = self.shared_learner(fea_img)
            fea_audio_c = self.shared_learner(fea_audio)
            # 跨模态交互
            mixed_fea = torch.cat((fea_text_c, fea_img_c, fea_audio_c), dim=1)
            mixed_fea = self.mix_learner(mixed_fea)
            fea = self.pool_learner(mixed_fea)

        output = self.classifier(fea)

        return output,fea



from utils.tools import *
import torch
from torch import nn
from model.ModalityModel import ModalityModel
import torch.nn.functional as F
  
class MultimodalLateFusion(nn.Module):
    def __init__(self, feat_dim=128, num_views=3, num_classes=2, tau=1.0, use_norm=True, fusion_type='adaptive'):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_views = num_views
        self.tau = tau
        self.use_norm = use_norm
        self.fusion_type = fusion_type

        self.gate = nn.Sequential(
            nn.Linear(feat_dim * num_views, 32),
            nn.ReLU(),
            nn.Linear(32, num_views),
            # nn.Softplus(),
        )

        if fusion_type == 'adaptive':
            self.classifier = nn.Linear(feat_dim * (num_views + 1), num_classes)
        elif fusion_type == 'concat':
            self.classifier = nn.Linear(feat_dim * num_views, num_classes)
        else:  # 'sum'
            self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, fea_list, logits_list):
        combined_feats = torch.cat(fea_list, dim=-1)  # [B, feat_dim*num_views]
        alpha = self.gate(combined_feats)  # [B, num_views]
        weights = F.softmax(alpha, dim=-1)

        weighted_feats = torch.stack(fea_list, dim=1)  # [B, num_views, feat_dim]

        # if self.use_norm:
        #     weighted_feats = F.normalize(weighted_feats, p=2, dim=-1)  # L2

        # reweight
        weighted_feats = weights.unsqueeze(-1) * weighted_feats  # [B, num_views, feat_dim]
        
        # fusion
        if self.fusion_type == 'concat':
            fused_fea = weighted_feats.flatten(start_dim=1)
        elif self.fusion_type == 'sum':
            fused_fea = torch.sum(weighted_feats, dim=1)
        else:  # 'adaptive'
            fused_fea = torch.cat([
                torch.sum(weighted_feats, dim=1),
                weighted_feats.flatten(start_dim=1)
            ], dim=-1)

        fused_logits = self.classifier(fused_fea)  # [B, num_classes]
        distill_loss = self.compute_loss(logits_list, alpha)
        return fused_logits,weights,distill_loss,fused_fea

    def compute_loss(self, logits_list, alpha):
        tau_pred = self.tau * 0.5
        tau_weight = self.tau
        
        view_entropy = []
        for logits in logits_list:
            prob = F.softmax(logits / tau_pred, dim=-1)
            entropy = -torch.sum(prob * torch.log(torch.clamp(prob, min=1e-6)), dim=-1)
            view_entropy.append(entropy)
        
        target_weights = F.softmax(-torch.stack(view_entropy, dim=1)/tau_weight, dim=-1)
        current_weights = F.softmax(alpha, dim=-1)
        
        # KL散度
        distill_loss = F.kl_div(
            F.log_softmax(target_weights, dim=-1),
            F.softmax(alpha, dim=-1),
            reduction='batchmean'
        )
        weight_entropy = -torch.sum(current_weights * torch.log(current_weights), dim=-1).mean()
        distill_loss += 0.05 * weight_entropy
        
        return distill_loss
    
    
class MDNet(torch.nn.Module):
    def __init__(self, dataset, model_path, module_deep, dropout):
        super(MDNet, self).__init__()
        self.dataset = dataset
        self.dropout = dropout
        self.module_deep = module_deep
        self.save_param_dir = model_path
        
        self.text_model = ModalityModel(dataset=self.dataset,modality='Text',module_deep=self.module_deep,dropout=self.dropout)
        self.visual_model = ModalityModel(dataset=self.dataset, modality='Visual',module_deep=self.module_deep,dropout=self.dropout)
        self.audio_model = ModalityModel(dataset=self.dataset, modality='Audio',module_deep=self.module_deep,dropout=self.dropout)
        self.mixed_model = ModalityModel(dataset=self.dataset, modality='Mixed',module_deep=self.module_deep,dropout=self.dropout)
        
        self.dynamic_logits_fusion = MultimodalLateFusion(feat_dim=128)

    def forward(self, **kwargs):
        text_logit,fea_text = self.text_model(**kwargs)
        visual_logit,fea_visual = self.visual_model(**kwargs)
        audio_logit,fea_audio = self.audio_model(**kwargs)
        mixed_logit,mixed_fea = self.mixed_model(**kwargs)

        fea_list = [fea_text,fea_visual,fea_audio,mixed_fea]
        logits_list = [text_logit,visual_logit,audio_logit,mixed_logit]
        
        final_logit,weights,distill_loss,fused_fea = self.dynamic_logits_fusion(fea_list, logits_list)
        return final_logit,distill_loss,logits_list



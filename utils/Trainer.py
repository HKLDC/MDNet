import time
from tqdm import tqdm
from utils.metrics import *
import torch
from torch import nn
from collections import defaultdict  
import torch.nn.functional as F

def joint_loss(branch_logits, total_logits, targets, distill_loss):
    assert len(branch_logits) == 4, "need 4 logits"
    
    criterion = nn.CrossEntropyLoss()

    text_logits, visual_logits, audio_logits, mixed_logits = branch_logits
    text_loss = criterion(text_logits, targets)
    visual_loss = criterion(visual_logits, targets)
    audio_loss = criterion(audio_logits, targets)
    mixed_loss = criterion(mixed_logits, targets)
    loss = criterion(total_logits, targets)

    joint_loss = (
        (text_loss + visual_loss + audio_loss + mixed_loss) / 4
        + 1.0 * loss
        + 1.0 * distill_loss
    )
    
    return joint_loss, text_loss, visual_loss, audio_loss, mixed_loss, loss


import json
import os
import numpy as np
from datetime import datetime

class GradientModulatorLogger:
    def __init__(self, modulator, log_dir='modulator_logs'):
        self.modulator = modulator
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.log_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'branch_names': modulator.branch_names
            },
            'epochs': []
        }

    def log_epoch(self, epoch, final_loss):
        epoch_data = {
            'epoch': epoch,
            'branch_data': {}
        }

        for name in self.modulator.branch_names:
            train_loss = self.modulator.train_losses.get(name, [np.nan])[-1]
            val_loss = self.modulator.dev_losses.get(name, [np.nan])[-1]
            
            epoch_data['branch_data'][name] = {
                'train_loss': float(train_loss),
                'val_loss': float(val_loss)
            }

        if epoch >= 2:
            for name in self.modulator.branch_names:
                train_loss = self.modulator.train_losses.get(name, [np.nan])[-1]
                val_loss = self.modulator.dev_losses.get(name, [np.nan])[-1]
                OG = self.modulator.OG_ratio_history.get(name, [1.0])[-1]
                scale = self.modulator.scale_history.get(name, [1.0])[-1]
                true_scale = self.modulator.true_scale_history.get(name, [1.0])[-1]
                
                epoch_data['branch_data'][name] = {
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                    'OG': float(OG),
                    'scale': float(scale),
                    'true_scale': float(true_scale)
                }
                
        epoch_data['branch_data']['final'] = {
                'train_loss': float(train_loss),
                'val_loss': float(final_loss),
        }
        
        self.log_data['epochs'].append(epoch_data)
        
    def save_logs(self):
        """save logs to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"modulator_log_{timestamp}.json"
        path = os.path.join(self.log_dir, filename)
        
        with open(path, 'w') as f:
            json.dump(self.log_data, f, indent=2)
        return path

class FourBranchGradientModulator:
    def __init__(self, min_epochs=3, super_epoch=1):
        self.branch_names = ["text", "visual", "audio", "mixed"]
        self.min_epochs = min_epochs
        self.super_epoch = super_epoch

        self.train_losses = defaultdict(list)
        self.dev_losses = defaultdict(list)

        self.G_history = {name: [] for name in self.branch_names}
        self.O_history = {name: [] for name in self.branch_names}
        self.OG_ratio_history = {name: [] for name in self.branch_names}
        self.scale_history = {name: [] for name in self.branch_names}
        self.true_scale_history = {name: [] for name in self.branch_names}  # 新增

        # super epoch
        self.current_super_epoch = 0
        self.epoch_scales_buffer = {name: [] for name in self.branch_names}
        self.final_scales = {name: 0 for name in self.branch_names}

    def update(self, phase, losses):
        assert phase in ['train', 'val']
        assert len(losses) == 4, "need 4 branch losses"

        for i, name in enumerate(self.branch_names):
            if phase == 'train':
                self.train_losses[name].append(losses[i])
            else:
                self.dev_losses[name].append(losses[i])

    def _compute_G(self, branch_name):
        """G = |L_val[N] - L_val[0]|"""
        dev_losses = self.dev_losses[branch_name]
        return abs(dev_losses[-1] - dev_losses[0])

    def _compute_O(self, branch_name):
        """O = ||gap_now| - |gap_prev||"""
        train_losses = self.train_losses[branch_name]
        dev_losses = self.dev_losses[branch_name]

        current_gap = abs(dev_losses[-1] - train_losses[-1])
        prev_gap = abs(dev_losses[0] - train_losses[0])
        return abs(current_gap - prev_gap)

    def _compute_GO_ratio(self, branch_name):
        """|G/O| """
        G = self._compute_G(branch_name)
        O = self._compute_O(branch_name)
        return G / O

    def compute_scales(self):
        epoch_scales = self._compute_scales_for_epoch()

        for i, name in enumerate(self.branch_names):
            self.epoch_scales_buffer[name].append(epoch_scales[i])

        if len(self.dev_losses['text']) < self.min_epochs:
            return [self.final_scales[name] for name in self.branch_names]

        if len(self.dev_losses['text']) % self.super_epoch == 0:
            for name in self.branch_names:
                if len(self.epoch_scales_buffer[name]) > 0:
                    self.final_scales[name] = np.mean(self.epoch_scales_buffer[name])
            self.epoch_scales_buffer = {name: [] for name in self.branch_names}
            self.current_super_epoch += 1

        for name in self.branch_names:
            self.true_scale_history[name].append(self.final_scales[name])

        return [self.final_scales[name] for name in self.branch_names]

    def sigmoid(self, x, c, s=2.0):
        return 1 / (1 + np.exp(-s * (x - c)))

    def _compute_scales_for_epoch(self):
        scales = []
        valid_ratios = []
        for name in self.branch_names:
            ratios = self.OG_ratio_history.get(name, [])
            if len(ratios) >= 3:
                q25, q75 = np.percentile(ratios[-5:], [25, 75])
                valid_ratios.extend([x for x in ratios[-5:] if q25 <= x <= q75])
        baseline_OG = np.mean(valid_ratios) if valid_ratios else 1.0

        for name in self.branch_names:
            G = self._compute_G(name)
            O = self._compute_O(name)
            OG_ratio = self._compute_GO_ratio(name)

            self.G_history[name].append(G)
            self.O_history[name].append(O)
            self.OG_ratio_history[name].append(OG_ratio)

            # relative_ratio
            scale = self.sigmoid(OG_ratio,baseline_OG)
            self.scale_history[name].append(scale)

            scales.append(float(scale))

        return scales


class GradientMonitor:
    @staticmethod
    def get_gradient_stats(model):
        stats = {}
        branches = {
            "text": model.text_model,
            "visual": model.visual_model,
            "audio": model.audio_model,
            "mixed": model.mixed_model,
            "fusion": model.dynamic_logits_fusion
        }
        
        for name, module in branches.items():
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            if not grads:
                stats[name] = {"mean": 0, "std": 0, "norm": 0}
                continue
                
            stacked = torch.stack([g.abs().mean() for g in grads])
            stats[name] = {
                "mean": stacked.mean().item(),
                "std": stacked.std().item(),
                "norm": torch.norm(torch.stack([g.norm() for g in grads])).item()
            }
        return stats

    @staticmethod
    def log_gradient_diff(before, after, prefix=""):
        diff = {}
        for k in before.keys():
            diff[k] = {
                "mean_diff": (after[k]["mean"] - before[k]["mean"]) / (before[k]["mean"] + 1e-8),
                "norm_ratio": after[k]["norm"] / (before[k]["norm"] + 1e-8)
            }
        print(f"{prefix} Gradient Changes:")
        for k, v in diff.items():
            print(f"  {k.upper():<6} - MeanΔ: {v['mean_diff']:+.2%}  NormRatio: {v['norm_ratio']:.2f}x")

class Trainer():
    def __init__(self,
                model,
                 device,
                 lr,
                 dropout,
                 dataloaders,
                 weight_decay,
                 save_param_path,
                 writer, 
                 epoch_stop,
                 epoches,
                 save_threshold = 0.0, 
                 start_epoch = 0,
                 ):
        
        self.model = model
        self.device = device
        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.epoch_stop = epoch_stop
        self.save_threshold = save_threshold
        self.writer = writer
        

        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            self.save_param_path = os.makedirs(save_param_path)
            self.save_param_path= save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
    
        self.criterion = nn.CrossEntropyLoss()

    def _apply_gradient_scales(self, scales):
        branch_models = [
            self.model.text_model,
            self.model.visual_model,
            self.model.audio_model,
            self.model.mixed_model
        ]

        for model, scale in zip(branch_models, scales):
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = param.grad * (scale+1)

        for param in self.model.dynamic_logits_fusion.parameters():
            if param.grad is not None:
                param.grad *= 1.0  # 保持
                    
    def train(self):
        since = time.time()
        self.model.cuda()
        best_acc_val = 0.0
        best_epoch_val = 0
        is_earlystop = False
        last_save_path = ''

        modulator = FourBranchGradientModulator()
        logger = GradientModulatorLogger(modulator)
        
        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch+1, self.start_epoch+self.num_epochs))
            print('-' * 50)

            p = float(epoch) / 100
            lr = self.lr / (1. + 10 * p) ** 0.75
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
            
            for phase in ['train', 'val', 'test']:
                if phase == 'train':
                    self.model.train()
                    if epoch > 1:
                        scales = modulator.compute_scales()
                        # print(f"Epoch {epoch}: Gradient Scales - Text={scales[0]:.2f}, Visual={scales[1]:.2f}, "
              # f"Audio={scales[2]:.2f}, Mixed={scales[3]:.2f}")
                else:
                    self.model.eval()
                print('-' * 10)
                print (phase.upper())
                print('-' * 10)

                running_loss = 0.0 
                text_running_loss = 0.0
                visual_running_loss = 0.0
                audio_running_loss = 0.0
                mixed_running_loss = 0.0 
                
                tpred = []
                tlabel = []
                    
                for batch in tqdm(self.dataloaders[phase]):
                    batch_data=batch
                    # to gpu
                    for k,v in batch_data.items():
                        batch_data[k]=v.cuda()
                    label = batch_data['label']

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs,distill_loss,logits_list = self.model(**batch_data)
                        _, preds = torch.max(outputs, 1)

                        total_loss, text_loss, visual_loss, audio_loss, mixed_loss, loss = joint_loss(branch_logits=logits_list, total_logits=outputs, targets=label, distill_loss=distill_loss)

                        if phase == 'train':
                            self.optimizer.zero_grad()
                            total_loss.backward()

                            pre_grad = GradientMonitor.get_gradient_stats(self.model)
                            if epoch >= 3:
                                self._apply_gradient_scales(scales)
                            post_grad = GradientMonitor.get_gradient_stats(self.model)
                            GradientMonitor.log_gradient_diff(pre_grad, post_grad, f"Epoch {epoch}")
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()

                    tlabel.extend(label.detach().cpu().numpy().tolist())
                    tpred.extend(preds.detach().cpu().numpy().tolist())
                    
                    running_loss += loss.item() * label.size(0)
                    text_running_loss += text_loss.item() * label.size(0)
                    visual_running_loss += visual_loss.item() * label.size(0)
                    audio_running_loss += audio_loss.item() * label.size(0)
                    mixed_running_loss += mixed_loss.item() * label.size(0)
                
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                text_epoch_loss = text_running_loss / len(self.dataloaders[phase].dataset)
                visual_epoch_loss = visual_running_loss / len(self.dataloaders[phase].dataset)
                audio_epoch_loss = audio_running_loss / len(self.dataloaders[phase].dataset)
                mixed_epoch_loss = mixed_running_loss / len(self.dataloaders[phase].dataset)
                losses = [text_epoch_loss,visual_epoch_loss,audio_epoch_loss,mixed_epoch_loss]
                
                # 输出final result
                print('Loss: {:.4f} '.format(epoch_loss))
                final_results = metrics(tlabel, tpred)
                print(final_results)
                # get_confusionmatrix_fnd(tpred,tlabel)
                
                if phase == 'val' and final_results['acc'] > best_acc_val:
                    best_acc_val = final_results['acc']
                    best_epoch_val = epoch + 1
                    if best_acc_val > self.save_threshold:
                        if os.path.exists(last_save_path):
                            print('delete the previous checkpoint...')
                            os.remove(last_save_path)
                        save_path = self.save_param_path + "_val_epoch" + str(best_epoch_val) + "_{0:.4f}".format(best_acc_val)
                        # torch.save(self.model.state_dict(),save_path)
                        # last_save_path = save_path
                        # print("saved " + self.save_param_path + "_test_epoch" + str(
                        #     best_epoch_val) + "_{0:.4f}".format(best_acc_val))
                    else:
                        if epoch - best_epoch_val >= self.epoch_stop - 1:
                            is_earlystop = True
                            print("early stopping...")

                if phase == 'train' or phase == 'val':
                    for i, branch in enumerate(["text", "visual", "audio", "mixed"]):
                        # tracker.update(phase, losses)
                        modulator.update(phase, losses)
                if phase == 'val':
                    logger.log_epoch(epoch,epoch_loss)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best model on val: epoch" + str(best_epoch_val) + "_" + str(best_acc_val))
        log_path = logger.save_logs()
        print(f"Logs saved to: {log_path}")
        return True

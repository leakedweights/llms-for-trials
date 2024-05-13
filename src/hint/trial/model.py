import torch 
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam

import matplotlib.pyplot as plt
from functools import reduce
from sklearn.metrics import f1_score, auc, precision_recall_curve, roc_auc_score, confusion_matrix
import numpy as np
from sklearn.utils import resample

from copy import deepcopy 
from tqdm import tqdm
import pickle

from .layers import FeedForward

from typing import Optional, Any


class TrialModel(nn.Module):
    def __init__(self,
                 toxicity_encoder: nn.Module,
                 disease_encoder: nn.Module,
                 protocol_embedding_size: int,
                 embedding_size: int,
                 num_ffn_layers: int,
                 num_pred_layers: int,
                 phase_dim: int,
                 device: Any,
                 ablations: dict = {"config": {"base_model": True}},
                 name: str = "trial_model"):
        
        super(TrialModel, self).__init__()
        
        self.toxicity_encoder = toxicity_encoder
        self.disease_encoder = disease_encoder
        self.embedding_size = embedding_size
        self.ablations = ablations
        self.model_name = name
        self.include_all = ablations["config"].get("base_model", False)
        self.device = device
                
        encoder_dim = toxicity_encoder.embedding_size + disease_encoder.embedding_size + protocol_embedding_size + embedding_size
                
        self.phase_encoder = FeedForward(
            input_dim=phase_dim,
            hidden_dim=embedding_size,
            output_dim=embedding_size,
            num_layers=num_ffn_layers
        )
            
        self.multimodal_encoder = FeedForward(
            input_dim=encoder_dim,
            hidden_dim=embedding_size,
            output_dim=embedding_size,
            num_layers=num_ffn_layers
        )
        
        self.disease_risk_encoder = FeedForward(
            input_dim=disease_encoder.embedding_size,
            output_dim=embedding_size,
            hidden_dim=embedding_size,
            num_layers=num_ffn_layers
        )
        
        self.interaction_encoder = FeedForward(
                        input_dim=2*embedding_size,
                        hidden_dim=embedding_size,
                        output_dim=embedding_size,
                        num_layers=num_ffn_layers
        )
        
        self.pk_encoder = FeedForward(
            input_dim=toxicity_encoder.embedding_size,
            output_dim=embedding_size,
            hidden_dim=5*embedding_size,
            num_layers=2*num_ffn_layers
        )
        
        self.trial_encoder = FeedForward(
            input_dim=2*embedding_size,
            output_dim=embedding_size,
            hidden_dim=embedding_size,
            num_layers=num_ffn_layers
        )
        
        self.pred = nn.Sequential(
            FeedForward(input_dim=embedding_size,
                        hidden_dim=embedding_size,
                        output_dim=1,
                        num_layers=num_pred_layers)
        )
    
    def forward(self, smiles, icd, protocol_embedding, phase):
        icd_embedding = self.disease_encoder.forward_code_lst3(icd)
        molecule_embedding = self.toxicity_encoder(smiles.to(self.device)).squeeze()                
        phase_embedding = self.phase_encoder(phase)
                
        encoder_embedding = self.multimodal_encoder(torch.cat([
            molecule_embedding,
            icd_embedding,
            protocol_embedding,
            phase_embedding
        ], 1)) 
        
        disease_risk_embedding = self.disease_risk_encoder(icd_embedding)
        interaction_embedding  = self.interaction_encoder(torch.cat([encoder_embedding, disease_risk_embedding], 1))
        pk_embedding = self.pk_encoder(molecule_embedding)
        trial_embedding = self.trial_encoder(torch.cat([interaction_embedding, pk_embedding], 1))
        
        embedding_options = {
            "disease_embedding": icd_embedding,
            "molecule_embedding": molecule_embedding,
            "protocol_embedding": protocol_embedding,
            "encoder_embedding": encoder_embedding,
            "disease_risk_embedding": disease_risk_embedding,
            "interaction_embedding": interaction_embedding,
            "pk_embedding": pk_embedding,
            "trial_embedding": trial_embedding
        }

        for key, value in embedding_options.items():
            if self.ablations["config"].get(key, self.include_all):
                x = value
            
        output = self.pred(x)
            
        return output
    
class Trainer:
    def __init__(self, model: TrialModel, weight_decay: float, lr: float, device: Any):
        self.model = model
        self.device = device
        self.optimizer = Adam(self.model.parameters(), weight_decay=weight_decay, lr=lr)

    def train(self, epochs: int, train_dataloader: DataLoader, valid_dataloader: DataLoader, test_dataloader: DataLoader):
        self.model.to(self.device)

        valid_loss, _ = self.evaluate(valid_dataloader, return_loss=True)
        best_valid_loss = valid_loss
        best_model = deepcopy(self.model)

        train_losses = []
        valid_losses = [valid_loss]

        for epoch in range(epochs):
            epoch_losses = []
            for nctids, labels, smiles, icdcodes, criteria, phase in tqdm(train_dataloader):
                labels = labels.to(self.device)
                smiles = smiles.to(self.device)
                phase = phase.to(self.device)
                criteria = criteria.to(self.device)
                outputs = self.model.forward(smiles, icdcodes, criteria, phase).view(-1)
                loss = bce_loss(outputs, labels.float())
                epoch_losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            train_losses.append(avg_epoch_loss)

            valid_loss, _ = self.evaluate(valid_dataloader, return_loss=True)
            valid_losses.append(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model = deepcopy(self.model)

        self.model = deepcopy(best_model)
        eval_metrics = self.evaluate(test_dataloader, return_loss=False)

        return eval_metrics
    
    def evaluate(self, dataloader: DataLoader, return_loss: bool = False):
        self.model.eval()
        with torch.no_grad():
            all_predictions = []
            all_labels = []
            total_loss = 0
            for nctids, labels, smiles, icdcodes, criteria, phase in dataloader:
                labels = labels.to(self.device)
                smiles = smiles.to(self.device)
                phase = phase.to(self.device)
                criteria = criteria.to(self.device)
                
                outputs = self.model.forward(smiles, icdcodes, criteria, phase).view(-1)

                if return_loss:
                    loss = bce_loss(outputs, labels.float())
                    total_loss += loss.item()

                predictions = (outputs > 0.5).float()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
            
            tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
            f1 = f1_score(all_labels, all_predictions)
            
            precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
            pr_auc = auc(recall, precision)
            
            roc_auc = roc_auc_score(all_labels, all_predictions)
            
            self.model.train()
            
            metrics = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'f1': f1, 'pr_auc': pr_auc, 'roc_auc': roc_auc}
            
            if return_loss:
                return total_loss / len(dataloader), metrics
            else:
                return metrics
            
    def test(self, test_dataloader: DataLoader):
        self.model.eval()
        with torch.no_grad():
            phase_metrics = defaultdict(list)

            for nctids, labels, smiles, icdcodes, criteria, phase in test_dataloader:
                labels = labels.to(self.device)
                smiles = smiles.to(self.device)
                phase = phase.to(self.device)
                criteria = criteria.to(self.device)

                outputs = self.model.forward(smiles, icdcodes, criteria, phase).view(-1)
                predictions = (outputs > 0.5).float()

                for ph, lbl, pred in zip(phase.cpu().numpy(), labels.cpu().numpy(), predictions.cpu().numpy()):
                    phase_metrics[ph].append((lbl, pred))

            for ph, results in phase_metrics.items():
                labels, predictions = zip(*results)
                tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
                f1 = f1_score(labels, predictions)
                precision, recall, _ = precision_recall_curve(labels, predictions)
                pr_auc = auc(recall, precision)
                roc_auc = roc_auc_score(labels, predictions)

                print(f"--- Phase {ph} Metrics ---")
                print(f"Accuracy: {(tp+tn)/(tp+tn+fp+fn):.3f}, TP: {tp}, FP:{fp}, TN:{tn}, FN:{fn}")
                print(f"F1-Score: {f1:.3f}")
                print(f"ROC-AUC: {roc_auc:.3f}")
                print(f"PR-AUC: {pr_auc:.3f}")
                print("-"*50)

        self.model.train()
        
    def bootstrap_test(self, test_dataloader: DataLoader, sample_num: int = 20):
        self.model.eval()
        phase_metrics = defaultdict(list)

        with torch.no_grad():
            for _, labels, smiles, icdcodes, criteria, phase in test_dataloader:
                labels = labels.to(self.device)
                smiles = smiles.to(self.device)
                phase = phase.to(self.device)
                criteria = criteria.to(self.device)

                outputs = self.model.forward(smiles, icdcodes, criteria, phase).view(-1)
                predictions = outputs.sigmoid().cpu().numpy()
                labels = labels.cpu().numpy()
                phases = phase.cpu().numpy()

                for ph, lbl, pred in zip(phases, labels, predictions):
                    phase_metrics[ph].append((lbl, pred))

        bootstrap_results = defaultdict(lambda: defaultdict(list))

        for ph, results in phase_metrics.items():
            labels, predictions = zip(*results)
            labels = np.array(labels)
            predictions = np.array(predictions)

            for _ in range(sample_num):
                bs_labels, bs_predictions = resample(labels, predictions)

                precision, recall, _ = precision_recall_curve(bs_labels, bs_predictions)
                pr_auc = auc(recall, precision)
                roc_auc = roc_auc_score(bs_labels, bs_predictions)

                bs_predictions_binary = (bs_predictions > 0.5).astype(int)
                f1 = f1_score(bs_labels, bs_predictions_binary)
                accuracy = np.mean(bs_labels == bs_predictions_binary)

                bootstrap_results[ph]['pr_auc'].append(pr_auc)
                bootstrap_results[ph]['f1'].append(f1)
                bootstrap_results[ph]['roc_auc'].append(roc_auc)
                bootstrap_results[ph]['accuracy'].append(accuracy)

        self.model.train()

        for ph, metrics in bootstrap_results.items():
            print(f"--- Phase {ph} Metrics ---")
            for metric, values in metrics.items():
                print(f"{metric.upper()} - mean: {np.mean(values):.4f}, std: {np.std(values):.4f}")
            print("-"*50)

        return bootstrap_results

from dataclasses import dataclass
from typing import Dict, List, Union, Tuple

from loss_utils import pad_to_length

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LogSoftmaxLoss:
    labels_pad_value: int = -100
    
    def __call__(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        assert logits.shape[:-1] == labels.shape, "The shape of logits and labels don't match."
    
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != self.labels_pad_value)
        
        labels[labels == self.labels_pad_value] = 0
        
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        loss = -(per_token_logps * loss_mask).sum(-1).mean() 
        
        return loss
    
@dataclass
class DirectPreferenceOptimizationLoss:
    labels_pad_value: int = -100
    beta: float = 0.1
    label_smoothing: float = 0.0
    averatge_log_prob: bool = False
    
    def __call__(self, policy: nn.Module, reference: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> torch.FloatTensor:
        policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(policy, batch)
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(reference, batch)
        losses, chosen_rewards, rejected_rewards = self.preference_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        return losses.mean(), chosen_rewards, rejected_rewards, reward_accuracies
    
    def get_batch_logps(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            logits: Shape: (B, L, V)
            labels: Shape: (B, L)
        """
        assert logits.shape[:-1] == labels.shape, "The shape of logits and labels don't match."
        
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != self.labels_pad_value)
        
        labels[labels == self.labels_pad_value] = 0
        
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        
        if self.averatge_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1) #B,
        else:
            return (per_token_logps * loss_mask).sum(-1) #B,
    
    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
        concatenated_batch = {}
        for k in batch:
            if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
                pad_value = self.labels_pad_value if 'labels' in k else 0
                concatenated_key = k.replace('chosen', 'concatenated')
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
                pad_value = self.labels_pad_value if 'labels' in k else 0
                concatenated_key = k.replace('rejected', 'concatenated')
                concatenated_batch[concatenated_key] = torch.cat((
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ), dim=0)
        return concatenated_batch
    
    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        concatenated_batch = self.concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps = self.get_batch_logps(all_logits, concatenated_batch['concatenated_labels'])
        batch_split_idx = batch['chosen_input_ids'].shape[0]
        chosen_logps = all_logps[:batch_split_idx]
        rejected_logps = all_logps[batch_split_idx:]
        return chosen_logps, rejected_logps
                
    def preference_loss(self, 
                        policy_chosen_logps: torch.FloatTensor,
                        policy_rejected_logps: torch.FloatTensor,
                        reference_chosen_logps: torch.FloatTensor,
                        reference_rejected_logps: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        
        logits = pi_logratios - ref_logratios
        
        losses = -F.logsigmoid(self.beta*logits)*(1-self.label_smoothing) - F.logsigmoid(-self.beta*logits)*self.label_smoothing
        
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        
        return losses, chosen_rewards, rejected_rewards

@dataclass
class SimplePreferenceOptimizationLoss:
    labels_pad_value: int = -100
    beta: float = 2.0
    gamma_beta_ratio: float = 0.5
    label_smoothing: float = 0.0
    average_log_prob: bool = True
    
    def __call__(self, policy: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> torch.FloatTensor:
        policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(policy, batch)
        losses, chosen_rewards, rejected_rewards = self.preference_loss(policy_chosen_logps, policy_rejected_logps)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        return losses.mean(), chosen_rewards, rejected_rewards, reward_accuracies
    
    def get_batch_logps(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            logits: Shape: (B, L, V)
            labels: Shape: (B, L)
        """
        assert logits.shape[:-1] == labels.shape, "The shape of logits and labels don't match."
        
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != self.labels_pad_value)
        
        labels[labels == self.labels_pad_value] = 0
        
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        
        if self.average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1) #B,
        else:
            return (per_token_logps * loss_mask).sum(-1) #B,
    
    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
        concatenated_batch = {}
        for k in batch:
            if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
                pad_value = self.labels_pad_value if 'labels' in k else 0
                concatenated_key = k.replace('chosen', 'concatenated')
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
                pad_value = self.labels_pad_value if 'labels' in k else 0
                concatenated_key = k.replace('rejected', 'concatenated')
                concatenated_batch[concatenated_key] = torch.cat((
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ), dim=0)
        return concatenated_batch
    
    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        concatenated_batch = self.concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps = self.get_batch_logps(all_logits, concatenated_batch['concatenated_labels'])
        batch_split_idx = batch['chosen_input_ids'].shape[0]
        chosen_logps = all_logps[:batch_split_idx]
        rejected_logps = all_logps[batch_split_idx:]
        return chosen_logps, rejected_logps
                
    def preference_loss(self, 
                        policy_chosen_logps: torch.FloatTensor,
                        policy_rejected_logps: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        logits = pi_logratios - self.gamma_beta_ratio
        
        losses = -F.logsigmoid(self.beta*logits)*(1-self.label_smoothing) - F.logsigmoid(-self.beta*logits)*self.label_smoothing
        
        chosen_rewards = self.beta * policy_chosen_logps.detach()
        rejected_rewards = self.beta * policy_rejected_logps.detach()
        
        return losses, chosen_rewards, rejected_rewards
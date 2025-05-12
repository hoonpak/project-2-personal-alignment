from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from loss import LogSoftmaxLoss, LogSoftmaxLossWithScore, DirectPreferenceOptimizationLoss, DirectPreferenceOptimizationLossWithScore
from argparse import ArgumentParser, BooleanOptionalAction

def get_args():
    parser = ArgumentParser()
    
    parser.add_argument("--data_ver", type=str, default="v01.1")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B")

    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--policy_dtype", type=str, default="bfloat16")
    parser.add_argument("--ref_dtype", type=str, default="bfloat16")
    
    parser.add_argument("--batch_size", type= int, default=64)
    parser.add_argument("--max_epoch", type= int, default=1)
    parser.add_argument("--grad_accum_steps", type= int, default=64)
    parser.add_argument("--eval_every", type= int, default=2496)

    parser.add_argument("--max_grad_norm", type= int, default=10)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--dpo_lr", type= str, default=5e-6)
    parser.add_argument("--sft_lr", type= str, default=5e-6)

    parser.add_argument("--sft_v", type=str, default="standard")
    parser.add_argument("--dpo_v", type=str, default="standard")

    return parser.parse_args()

class SuperviseFineTuning:
    def __init__(self, args):
        self.args = args
        self.macro_batch_size = args.batch_size
        self.micro_batch_size = int(self.macro_batch_size//args.grad_accum_steps)
        self.loss_function = LogSoftmaxLoss(labels_pad_value=-100)
        
    def train(self,
              preference,
              policy,
              optimizer,
              train_dataloader,
              eval_dataloader,
              ):
        writer = SummaryWriter(log_dir=f"./runs/{self.args.model_name}/sft/{self.args.data_ver}/{self.args.sft_v}/{preference}")
        
        example_counter = 0
        
        train_loss = 0
        
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            policy.train()
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            
            example_counter += self.micro_batch_size
            logits = policy(input_ids=input_ids, attention_mask=attention_mask).logits.to(torch.float32)
            loss = self.loss_function(logits, labels)
            (loss / self.args.grad_accum_steps).backward()
            
            train_loss += loss / self.args.grad_accum_steps
            
            if example_counter%self.macro_batch_size == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
                writer.add_scalars('GPU_Usage', {'GPU_Usage':round(torch.cuda.memory_reserved()/1e+9, 2)}, example_counter)
                writer.add_scalars('train_loss', {'train_loss':train_loss.detach().cpu().item()}, example_counter)
                writer.flush()
                
                train_loss = 0
                
            if (example_counter%self.args.eval_every == 0) or (example_counter == self.micro_batch_size):
                policy.eval()
                with torch.no_grad():
                    eval_loss = 0
                    for eval_num, batch in enumerate(eval_dataloader):
                        input_ids = batch['input_ids'].cuda()
                        attention_mask = batch['attention_mask'].cuda()
                        labels = batch['labels'].cuda()
                        logits = policy(input_ids=input_ids, attention_mask=attention_mask).logits.to(torch.float32)
                        eval_loss += self.loss_function(logits, labels).detach().cpu().item()
                    eval_loss /= eval_num
                    writer.add_scalars('eval_loss', {'eval_loss':eval_loss}, example_counter)
                    writer.flush()

        save_dir = f"../save/{self.args.model_name}/sft/{self.args.data_ver}/{self.args.sft_v}/{preference}"
        os.makedirs(save_dir, exist_ok=True)
        
        policy.save_pretrained(save_dir, from_pt=True)

class DirectPreferenceOptimization:
    def __init__(self, args):
        self.args = args
        self.macro_batch_size = args.batch_size
        self.micro_batch_size = int(self.macro_batch_size//args.grad_accum_steps)
        self.dpo_loss_func = DirectPreferenceOptimizationLoss()
        
    def train(self, 
              preference,
              policy,
              reference,
              optimizer,
              train_dataloader,
              eval_dataloader,
              ):
        writer = SummaryWriter(log_dir=f"./runs/{self.args.model_name}/dpo/{self.args.data_ver}/{self.args.dpo_v}/{preference}")

        example_counter = 0
        
        train_loss = 0
        train_chosen_rewards = 0
        train_rejected_rewards = 0
        train_reward_accuracies = 0
        
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            policy.train()
            batch['chosen_input_ids'] = batch['chosen_input_ids'].cuda()
            batch['chosen_attention_mask'] = batch['chosen_attention_mask'].cuda()
            batch['chosen_labels'] = batch['chosen_labels'].cuda()
            batch['rejected_input_ids'] = batch['rejected_input_ids'].cuda()
            batch['rejected_attention_mask'] = batch['rejected_attention_mask'].cuda()
            batch['rejected_labels'] = batch['rejected_labels'].cuda()
            
            example_counter += self.micro_batch_size
            losses, chosen_rewards, rejected_rewards, reward_accuracies = self.dpo_loss_func(policy, reference, batch)
            (losses/ self.args.grad_accum_steps).backward()
            
            train_loss += (losses.detach()/self.args.grad_accum_steps).cpu().item()
            train_chosen_rewards += (chosen_rewards.detach().mean()/self.args.grad_accum_steps).cpu().item()
            train_rejected_rewards += (rejected_rewards.detach().mean()/self.args.grad_accum_steps).cpu().item()
            train_reward_accuracies += (reward_accuracies.detach().mean()/self.args.grad_accum_steps).cpu().item()
            
            if example_counter%self.macro_batch_size == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
                writer.add_scalars('GPU_Usage', {'GPU_Usage':round(torch.cuda.memory_reserved()/1e+9, 2)}, example_counter)
                writer.add_scalars('train_loss', {'train_loss':train_loss}, example_counter)
                writer.add_scalars('train_chosen_rewards', {'train_chosen_rewards':train_chosen_rewards}, example_counter)
                writer.add_scalars('train_rejected_rewards', {'train_rejected_rewards':train_rejected_rewards}, example_counter)
                writer.add_scalars('train_reward_accuracies', {'train_reward_accuracies':train_reward_accuracies}, example_counter)
                writer.flush()
                
                train_loss = 0
                train_chosen_rewards = 0
                train_rejected_rewards = 0
                train_reward_accuracies = 0
                
            if (example_counter%self.args.eval_every == 0) or (example_counter == self.micro_batch_size):
                policy.eval()
                with torch.no_grad():
                    eval_loss = 0
                    eval_chosen_rewards = 0
                    eval_rejected_rewards = 0
                    eval_reward_accuracies = 0
                    for eval_num, batch in enumerate(eval_dataloader):
                        batch['chosen_input_ids'] = batch['chosen_input_ids'].cuda()
                        batch['chosen_attention_mask'] = batch['chosen_attention_mask'].cuda()
                        batch['chosen_labels'] = batch['chosen_labels'].cuda()
                        batch['rejected_input_ids'] = batch['rejected_input_ids'].cuda()
                        batch['rejected_attention_mask'] = batch['rejected_attention_mask'].cuda()
                        batch['rejected_labels'] = batch['rejected_labels'].cuda()
                        
                        losses, chosen_rewards, rejected_rewards, reward_accuracies = self.dpo_loss_func(policy, reference, batch)
                        
                        eval_loss += losses.detach().cpu().item()
                        eval_chosen_rewards += chosen_rewards.detach().mean().cpu().item()
                        eval_rejected_rewards += rejected_rewards.detach().mean().cpu().item()
                        eval_reward_accuracies += reward_accuracies.detach().mean().cpu().item()
                        
                    eval_loss /= eval_num
                    eval_chosen_rewards /= eval_num
                    eval_rejected_rewards /= eval_num
                    eval_reward_accuracies /= eval_num
                    
                    writer.add_scalars('eval_loss', {'eval_loss':eval_loss}, example_counter)
                    writer.add_scalars('eval_chosen_rewards', {'eval_chosen_rewards':eval_chosen_rewards}, example_counter)
                    writer.add_scalars('eval_rejected_rewards', {'eval_rejected_rewards':eval_rejected_rewards}, example_counter)
                    writer.add_scalars('eval_reward_accuracies', {'eval_reward_accuracies':eval_reward_accuracies}, example_counter)
                    writer.flush()
                    
        save_dir = f"../save/{self.args.model_name}/dpo/{self.args.data_ver}/{self.args.dpo_v}/{preference}"
        os.makedirs(save_dir, exist_ok=True)

        policy.save_pretrained(save_dir, from_pt=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import Dataset

import json
import numpy as np
import pandas as pd
import random

from functools import partial

import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
from torch import optim
from torch.utils.data import DataLoader

import os, sys
sys.path.append("../src")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from loss_utils import sft_preprocess, collator
from train import SuperviseFineTuning, get_args

def filtering_none(dataset, preference):
    if preference == "base":
        return dataset
    else:
        new_dataset = []
        for data in dataset:
            if data['performance_score'][preference]:
                new_dataset.append(data)
        return new_dataset

if __name__ == "__main__":
    args = get_args()
    
    for k, v in vars(args).items():
        print(f"{k :<20}: {v}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    macro_batch_size = args.batch_size
    micro_batch_size = int(macro_batch_size//args.grad_accum_steps) #2
    
    print("macro_batch_size: ", macro_batch_size)
    print("micro_batch_size: ", micro_batch_size)
    
    # preparing training dataset
    if args.model_name.startswith("Qwen"):
        tokenizer = AutoTokenizer.from_pretrained(f"{args.model_name}")
        new_tokens = ["<|score|>"]
        tokenizer.add_tokens(new_tokens)
    elif args.model_name.startswith("meta-llama"):
        tokenizer = AutoTokenizer.from_pretrained("../llama_tokenizer")
    else:
        raise ValueError()

    user_preference = "base"
    print("#"*25," ", user_preference, " ", "#"*25)
    train_dataset = [json.loads(line) for line in open(f"../data/train/{args.model_name.split("/")[0]}_sft_train_dataset_{args.data_ver}.jsonl","r").readlines()]
    eval_dataset = [json.loads(line) for line in open(f"../data/val/{args.model_name.split("/")[0]}_sft_val_dataset_{args.data_ver}.jsonl","r").readlines()][:512]

    train_dataset = filtering_none(train_dataset, user_preference)
    eval_dataset = filtering_none(eval_dataset, user_preference)
    
    sft_preprocess_ = partial(sft_preprocess, tokenizer=tokenizer)

    train_dataset = Dataset.from_pandas(pd.DataFrame(data=train_dataset))
    eval_dataset = Dataset.from_pandas(pd.DataFrame(data=eval_dataset))
    
    train_dataset = train_dataset.map(sft_preprocess_, batched=False, remove_columns=['train_data_id',  "conv_id", "query" , "chosen", "chosen_score" , "rejected", "rejected_score", "user_id", "user_profile", "performance_score"])
    eval_dataset = eval_dataset.map(sft_preprocess_, batched=False, remove_columns=['val_data_id',  "conv_id", "query" , "chosen", "chosen_score" , "rejected", "rejected_score", "user_id", "user_profile", "performance_score"])
    
    sft_instance = SuperviseFineTuning(args)
        
    collator_ = partial(collator, pad_token_id=tokenizer.eos_token_id)
    train_dataloader = DataLoader(train_dataset, batch_size=micro_batch_size, collate_fn=collator_, drop_last=True, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=micro_batch_size, collate_fn=collator_, drop_last=True, shuffle=False)

    # preparing model
    policy_dtype = getattr(torch, args.policy_dtype)
    policy = AutoModelForCausalLM.from_pretrained(f"{args.model_name}", low_cpu_mem_usage=True, torch_dtype=policy_dtype)
    
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none", 
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(policy, peft_config).cuda()
    
    print("policy trainable parameters: ")
    policy.print_trainable_parameters()
    
    optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, policy.parameters()), lr=float(args.sft_lr))

    sft_instance.train(preference=user_preference, policy=policy, optimizer=optimizer,
                        train_dataloader=train_dataloader, eval_dataloader=eval_dataloader)
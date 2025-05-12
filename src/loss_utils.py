import torch
from torch.nn.utils.rnn import pad_sequence

from typing import Union

def sft_preprocess(sample, tokenizer, max_length: int = 768, max_prompt_length: int = 512):
    input_text = sample['query']
    labels = sample['chosen']
    
    input_tokens = tokenizer(input_text, add_special_tokens=False)
    label_tokens = tokenizer(labels, add_special_tokens=False)
    
    label_tokens['input_ids'].append(tokenizer.eos_token_id)
    label_tokens['attention_mask'].append(1)
    
    if len(input_tokens['input_ids']) + len(label_tokens['input_ids']) > max_length:
        input_tokens['input_ids'] = input_tokens['input_ids'][-max_prompt_length:]
        input_tokens['attention_mask'] = input_tokens['attention_mask'][-max_prompt_length:]
        
    if len(input_tokens['input_ids']) + len(label_tokens['input_ids']) > max_length:
        label_tokens['input_ids'] = label_tokens['input_ids'][:max_length-max_prompt_length]
        label_tokens['attention_mask'] = label_tokens['attention_mask'][:max_length-max_prompt_length]
    
    input_tokens['labels'] = [-100]*len(input_tokens['input_ids']) + label_tokens['input_ids']
    input_tokens['input_ids'] += label_tokens['input_ids']
    input_tokens['attention_mask'] += label_tokens['attention_mask']
    
    return input_tokens

def dpo_preprocess(sample, tokenizer, max_length: int = 768, max_prompt_length: int = 512):
    prompts = sample['query']
    chosen_labels = sample['chosen']
    rejected_labels = sample['rejected']
    
    prompt_tokens = tokenizer(prompts, add_special_tokens=False)
    chosen_label_tokens = tokenizer(chosen_labels, add_special_tokens=False)
    rejected_label_tokens = tokenizer(rejected_labels, add_special_tokens=False)
    
    chosen_label_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_label_tokens['attention_mask'].append(1)
    rejected_label_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_label_tokens['attention_mask'].append(1)    
    
    if len(prompt_tokens['input_ids']) + len(chosen_label_tokens['input_ids']) > max_length:
        prompt_tokens['input_ids'] = prompt_tokens['input_ids'][-max_prompt_length:]
        prompt_tokens['attention_mask'] = prompt_tokens['attention_mask'][-max_prompt_length:]
        
    if len(prompt_tokens['input_ids']) + len(chosen_label_tokens['input_ids']) > max_length:
        chosen_label_tokens['input_ids'] = chosen_label_tokens['input_ids'][:max_length-max_prompt_length]
        chosen_label_tokens['attention_mask'] = chosen_label_tokens['attention_mask'][:max_length-max_prompt_length]
        
    if len(prompt_tokens['input_ids']) + len(rejected_label_tokens['input_ids']) > max_length:
        rejected_label_tokens['input_ids'] = rejected_label_tokens['input_ids'][:max_length-max_prompt_length]
        rejected_label_tokens['attention_mask'] = rejected_label_tokens['attention_mask'][:max_length-max_prompt_length]
    
    input_tokens = {}
    
    input_tokens['chosen_labels'] = [-100]*len(prompt_tokens['input_ids']) + chosen_label_tokens['input_ids']
    input_tokens['chosen_input_ids'] = prompt_tokens['input_ids'] + chosen_label_tokens['input_ids']
    input_tokens['chosen_attention_mask'] = prompt_tokens['attention_mask'] + chosen_label_tokens['attention_mask']
    input_tokens['rejected_labels'] = [-100]*len(prompt_tokens['input_ids']) + rejected_label_tokens['input_ids']
    input_tokens['rejected_input_ids'] = prompt_tokens['input_ids'] + rejected_label_tokens['input_ids']
    input_tokens['rejected_attention_mask'] = prompt_tokens['attention_mask'] + rejected_label_tokens['attention_mask']
    
    return input_tokens

def collator(batch, pad_token_id):
    padded_batch = {}
    for key in batch[0].keys():
        if key in ["pref_score", "chosen_score", "rejected_score"]:
            padded_batch[key] = torch.FloatTensor([[ex[key]] for ex in batch])
            continue
        # batch는 pad 처리가 되지 않은 Dict[List[List]] 형태로 입력으로 들어옴.
        to_pad = [torch.LongTensor(ex[key]) for ex in batch]
        # to_pad = [ex[key] for ex in batch]
        if key.endswith('input_ids'):
            padding_value = pad_token_id
        elif key.endswith('attention_mask'):
            padding_value = 0
        elif key.endswith('labels'):
            padding_value = -100
        else:
            raise ValueError(f"Unexpected key in batch '{key}'")
        padded_batch[key] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
    
    return padded_batch

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value*torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)
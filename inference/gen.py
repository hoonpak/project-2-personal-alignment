import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel
from tqdm import tqdm

import os
import json
import random

from argparse import ArgumentParser, BooleanOptionalAction

def filtering_none(dataset, preference):
    if (preference == "base") or (preference == None):
        return dataset
    else:
        new_dataset = []
        for data in dataset:
            if data['performance_score'][preference]:
                new_dataset.append(data)
        return new_dataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--is_base", action=BooleanOptionalAction)
    parser.add_argument("--score", action=BooleanOptionalAction)
    parser.add_argument("--user_profile", action=BooleanOptionalAction)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--s", type=str, default="sft")
    parser.add_argument("--data_ver", type=str, default="v01.2")
    parser.add_argument("--sft_v", type=str, default="0")
    parser.add_argument("--dpo_v", type=str, default="0")
    parser.add_argument("--preference", type=str)
    parser.add_argument("--sample", type=int, default=512)
    parser.add_argument("--query_max_length", type=int, default=384)
    parser.add_argument("--generate_max_length", type=int, default=256)
    args = parser.parse_args()
    
    for k, v in vars(args).items():
        print(f"{k :<20}: {v}")
        
    torch.manual_seed(42)
    np.random.seed(42)
    set_seed(42)
    random.seed(42)
    
    tmp_samples = [json.loads(line) for line in open(f"../data/test/{args.model_name.split("/")[0]}_test_dataset_v01.2.jsonl", "r").readlines()][:args.sample]
    samples = filtering_none(tmp_samples, args.preference)

    if args.model_name.startswith("Qwen"):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        new_tokens = ["<|score|>"]
        tokenizer.add_tokens(new_tokens)
    elif args.model_name.startswith("meta-llama"):
        tokenizer = AutoTokenizer.from_pretrained("../llama_tokenizer")
    else:
        raise ValueError()
        
    if args.is_base:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    else:
        if args.s == "dpo":
            lora_adapter_path = f"../save/{args.model_name}/{args.s}/{args.data_ver}/{args.dpo_v}/{args.preference}"
        else:
            lora_adapter_path = f"../save/{args.model_name}/{args.s}/{args.data_ver}/{args.sft_v}/{args.preference}"
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base_model, lora_adapter_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    model = model.cuda()

    query_max_length = args.query_max_length
    generate_max_length = args.generate_max_length

    save_dir = "../data/output"
    if args.is_base:
        output_file_path = f"/{args.model_name}/{args.data_ver}/base"
        os.makedirs(save_dir+output_file_path, exist_ok=True)
        write_file = open(os.path.join(save_dir+output_file_path, "base.jsonl"), "w")
    else:
        if args.s == "dpo":
            output_file_path = f"/{args.model_name}/{args.data_ver}/{args.s}/{args.dpo_v}"
        else:
            output_file_path = f"/{args.model_name}/{args.data_ver}/{args.s}/{args.sft_v}"
        os.makedirs(save_dir+output_file_path, exist_ok=True)
        write_file = open(os.path.join(save_dir+output_file_path, f"{args.preference}.jsonl"), "w")
    
    with torch.no_grad():
        for sample in tqdm(samples, total=len(samples)):
            u_profile = sample['user_profile'].replace("<|score|><|im_start|>system\n", "").replace("<|im_end|>", "").replace("<|score|><|begin_of_text|><|start_header_id|>system<|end_header_id|>\n", "").replace("<|eot_id|>", "")
            sys_prompt=""
            if args.model_name.split("/")[0] == "Qwen":
                if (args.score):
                    sys_prompt += "<|score|>"
                if (args.user_profile):
                    sys_prompt += "<|im_start|>system\n"+u_profile+"<|im_end|>\n"
            elif args.model_name.split("/")[0] == "meta-llama":
                if (args.score):
                    sys_prompt += "<|score|>"
                sys_prompt += "<|begin_of_text|>"
                if (args.user_profile):
                    sys_prompt += "<|start_header_id|>system<|end_header_id|>\n"+u_profile+"<|eot_id|>"
            sys_tokens = tokenizer(sys_prompt, return_tensors="pt")
            
            input_tokens = tokenizer(sample['query'], return_tensors="pt")
            for key in input_tokens.keys():
                if input_tokens[key].shape[1] > query_max_length:
                    input_tokens[key] = torch.LongTensor(input_tokens[key][0][-query_max_length:]).unsqueeze(0)
            
            tmp_input_tokens = {
                                'input_ids':torch.concat((sys_tokens['input_ids'], input_tokens['input_ids']), dim=1),
                                'attention_mask':torch.concat((sys_tokens['attention_mask'], input_tokens['attention_mask']), dim=1),
                                }
            
            prompt_length = tmp_input_tokens['input_ids'].shape[1]
            
            for key in tmp_input_tokens:
                tmp_input_tokens[key] = tmp_input_tokens[key].to(torch.long).to("cuda:0")
            if (args.preference == 'base') or (args.is_base):
                output_1 = model.generate(**tmp_input_tokens, no_repeat_ngram_size = 5, num_return_sequences=7, temperature=1, 
                                          do_sample=True, max_new_tokens=generate_max_length, pad_token_id=tokenizer.eos_token_id).to("cpu")
                tmp_dict = {
                            "conv_id": sample['conv_id'],
                            "input": tokenizer.decode(tmp_input_tokens['input_ids'][0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip(),
                            }
                for idx in range(7):
                    tmp_dict[f'output_{idx}'] = tokenizer.decode(output_1[idx][prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
                    
            else:
                output_1 = model.generate(**tmp_input_tokens, no_repeat_ngram_size = 5, temperature=1, do_sample=True,
                                        max_new_tokens=generate_max_length, pad_token_id=tokenizer.eos_token_id).to("cpu")
                tmp_dict = {
                            "conv_id": sample['conv_id'],
                            "input": tokenizer.decode(tmp_input_tokens['input_ids'][0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip(),
                            "output_temp_1": tokenizer.decode(output_1[0][prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
                            }
            
            write_file.write(json.dumps(tmp_dict, ensure_ascii=False) + "\n")

    write_file.close()
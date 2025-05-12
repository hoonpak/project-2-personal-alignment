import json
import time

from openai import OpenAI

from argparse import ArgumentParser, BooleanOptionalAction
from transformers import AutoTokenizer

import sys, os
sys.path.append("../src/")
sys.path.append("../utils")
from prompt import *

def get_base_request(write_file, dataset, test_dataset, tokenizer, concentrate_model):
    for idx, sample in enumerate(dataset):
        profile = test_dataset[idx]['user_profile']
        query = tokenizer.decode(tokenizer.encode(test_dataset[idx]['query'])[-384:]).strip()
        response_0 = sample['output_0']
        response_1 = sample['output_1']
        response_2 = sample['output_2']
        response_3 = sample['output_3']
        response_4 = sample['output_4']
        response_5 = sample['output_5']
        response_6 = sample['output_6']
        
        user_prompt = mk_base_mix_prompt(profile, query, response_0, response_1, response_2,
                                         response_3, response_4, response_5, response_6)
        tmp_request = mk_normal_request(test_dataset[idx]['conv_id'], seed=42,
                                        model=concentrate_model, user_prompt=user_prompt)
        write_file.write(json.dumps(tmp_request, ensure_ascii=False)+"\n")

def get_SPECTRUM_request(write_file, test_dataset, tokenizer, concentrate_model):
    for sample in test_dataset:
        profile = sample['user_profile']
        query = tokenizer.decode(tokenizer.encode(sample['query'])[-384:]).strip()
        user_prompt = mk_SPECTRUM_prompt(profile, query, sample['creativity'], sample['diversity'], sample['factuality'],
                                         sample['fluency'], sample['helpfulness'], sample['safety'], sample['values'])
        tmp_request = mk_normal_request(sample['conv_id'], seed=42, model=concentrate_model, user_prompt=user_prompt)
        write_file.write(json.dumps(tmp_request, ensure_ascii=False)+"\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--is_base", action=BooleanOptionalAction)
    parser.add_argument("--s", type=str, default="sft")
    parser.add_argument("--data_ver", type=str, default="v01.2")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--sft_v", type=str)
    parser.add_argument("--dpo_v", type=str)
    parser.add_argument("--preference", type=str)
    parser.add_argument("--con_model", type=str)
    args = parser.parse_args()
    
    for k, v in vars(args).items():
        print(f"{k :<20}: {v}")
    
    if args.model_name.startswith("Qwen"):
        tokenizer = AutoTokenizer.from_pretrained(f"{args.model_name}")
        new_tokens = ["<|score|>"]
        tokenizer.add_tokens(new_tokens)
    elif args.model_name.startswith("meta-llama"):
        tokenizer = AutoTokenizer.from_pretrained("../llama_tokenizer")
    
    gen_output_dir = f"../data/output"
    test_dataset = [json.loads(line) for line in open("../data/test/test_dataset_v01.2.jsonl", "r").readlines()]
    
    if (args.is_base) and (args.preference != "base"):
        response_path = os.path.join(gen_output_dir,f"{args.model_name}/{args.data_ver}/base/base.jsonl")
        dataset = [json.loads(line) for line in open(response_path, "r").readlines()]
        request_output_path = f"../data/requests/combine/{args.model_name}/{args.data_ver}/base"

    elif (not args.is_base) and (args.preference == "base"):
        if args.s == "sft":
            response_path = os.path.join(gen_output_dir,f"{args.model_name}/{args.data_ver}/{args.s}/{args.sft_v}/base.jsonl")
        elif args.s == "dpo":
            response_path = os.path.join(gen_output_dir,f"{args.model_name}/{args.data_ver}/{args.s}/{args.dpo_v}/base.jsonl")
        dataset = [json.loads(line) for line in open(response_path, "r").readlines()]
        if args.s == "sft":
            request_output_path = f"../data/requests/combine/{args.model_name}/{args.data_ver}/{args.s}/{args.sft_v}"
        elif args.s == "dpo":
            request_output_path = f"../data/requests/combine/{args.model_name}/{args.data_ver}/{args.s}/{args.dpo_v}"
            
    elif (not args.is_base) and (args.preference == "all"):
        for pref in ["values", "fluency", "factuality", "safety", "diversity", "creativity", "helpfulness"]:
            if args.s == "sft":
                response_path = os.path.join(gen_output_dir,f"{args.model_name}/{args.data_ver}/{args.s}/{args.sft_v}/{pref}.jsonl")
            elif args.s == "dpo":
                response_path = os.path.join(gen_output_dir,f"{args.model_name}/{args.data_ver}/{args.s}/{args.dpo_v}/{pref}.jsonl")
            dataset = [json.loads(line) for line in open(response_path, "r").readlines()]
            dataset_index = [data['conv_id'] for data in dataset]
            for sample in test_dataset:
                if sample['conv_id'] in dataset_index:
                    sample[pref] = dataset[dataset_index.index(sample['conv_id'])]['output_temp_1']
                else:
                    sample[pref] = "None"
        if args.s == "sft":
            request_output_path = f"../data/requests/combine/{args.model_name}/{args.data_ver}/{args.s}/{args.sft_v}"
        elif args.s == "dpo":
            request_output_path = f"../data/requests/combine/{args.model_name}/{args.data_ver}/{args.s}/{args.dpo_v}"

    os.makedirs(request_output_path, exist_ok=True)
    
    with open(request_output_path+f"/{args.con_model}_request.jsonl", "w") as write_file:
        if (args.is_base) or (args.preference == "base"):
            get_base_request(write_file, dataset, test_dataset, tokenizer, args.con_model)
        elif (args.preference == "all"):
            get_SPECTRUM_request(write_file, test_dataset, tokenizer, args.con_model)
    
    client = OpenAI(api_key=open("../utils/openai_key", "r").read().strip())
    input_file = client.files.create(
        file=open(request_output_path+f"/{args.con_model}_request.jsonl", "rb"),
        purpose="batch"
    )

    input_file_id = input_file.id
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    i = 0
    st_time = time.time()
    while True:
        batch_info = client.batches.retrieve(batch.id)
        print(f"* status: {batch_info.status:>13}  (elapsed_time... {time.time() - st_time:.2f} sec)")
        if batch_info.status == 'completed':
            cur_gen_file_id = batch_info.output_file_id
            file_response = client.files.content(cur_gen_file_id)
            output_file = open(request_output_path+f"/{args.con_model}_output.jsonl", "w")
            for i in file_response.iter_lines():
                output_file.write(json.dumps(json.loads(i), ensure_ascii=False) + "\n")
            output_file.close()
            print("END")
            break
        else:
            time.sleep(30)
            i += 1
            if i > 2880:
                print("The data didn't generate")
                break
            else:
                continue

    print("\nDONE.")
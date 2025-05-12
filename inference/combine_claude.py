import json
import time

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from argparse import ArgumentParser, BooleanOptionalAction
from transformers import AutoTokenizer

import sys, os
sys.path.append("../src/")
sys.path.append("../utils")
from prompt import *

def mk_claude_message(custom_id, concentrate_model, user_prompt):
    return Request(
        custom_id=custom_id,
        params=MessageCreateParamsNonStreaming(
            model=concentrate_model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": user_prompt
            }]
        )
    )

def get_base_request(requests, dataset, test_dataset, tokenizer, concentrate_model):
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
        requests.append(mk_claude_message(test_dataset[idx]['conv_id'], concentrate_model, user_prompt))
        
def get_SPECTRUM_request(requests, test_dataset, tokenizer, concentrate_model):
    for sample in test_dataset:
        profile = sample['user_profile']
        query = tokenizer.decode(tokenizer.encode(sample['query'])[-384:]).strip()
        user_prompt = mk_SPECTRUM_prompt(profile, query, sample['creativity'], sample['diversity'], sample['factuality'],
                                         sample['fluency'], sample['helpfulness'], sample['safety'], sample['values'])
        requests.append(mk_claude_message(sample['conv_id'], concentrate_model, user_prompt))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--is_base", action=BooleanOptionalAction)
    parser.add_argument("--s", type=str)
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

    requests = []
    if (args.is_base) or (args.preference == "base"):
        get_base_request(requests, dataset, test_dataset, tokenizer, args.con_model)
    elif (not args.is_base) and (args.preference == "all"):
        get_SPECTRUM_request(requests, test_dataset, tokenizer, args.con_model)
    
    client = anthropic.Anthropic(api_key=open("../utils/anthropic_key", "r").read().strip())
    
    message_batch = client.messages.batches.create(requests=requests)
    batch_id = message_batch.id
    i = 0
    st_time = time.time()
    while True:
        retrieved_batch = client.messages.batches.retrieve(batch_id)
        print(f"Batch {batch_id} processing status is {retrieved_batch.processing_status} (elapsed_time... {time.time() - st_time:.2f} sec)")
        if retrieved_batch.processing_status == 'ended':
            output_file = open(request_output_path+f"/{args.con_model}_output.jsonl", "w")
            for result in client.messages.batches.results(batch_id):
                output_file.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            output_file.close()
            print("END")
            break
        elif retrieved_batch.processing_status == "canceling":
            print("SOMETHINGS WRONG..")
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
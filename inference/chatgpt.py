import json
import time

from openai import OpenAI

from argparse import ArgumentParser, BooleanOptionalAction
from transformers import AutoTokenizer

import sys, os
sys.path.append("../src/")
sys.path.append("../utils")
from prompt import *

def get_request(write_file, test_dataset, tokenizer, args):
    for sample in test_dataset:
        profile = sample['user_profile']
        query = tokenizer.decode(tokenizer.encode(sample['query'])[-384:]).strip()
        
        if args.w_profile:
            user_prompt = mk_llm_w_profile_prompt(profile, query)
        else:
            user_prompt = mk_llm_wo_profile_prompt(query)
            
        tmp_request = mk_normal_request(sample['conv_id'], seed=42, model=args.model, user_prompt=user_prompt)
        write_file.write(json.dumps(tmp_request, ensure_ascii=False)+"\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--w_profile", action=BooleanOptionalAction)
    parser.add_argument("--data_ver", type=str, default="v01.2")
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    
    for k, v in vars(args).items():
        print(f"{k :<20}: {v}")
        
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    new_tokens = ["<|score|>"]
    tokenizer.add_tokens(new_tokens)
        
    test_dataset = [json.loads(line) for line in open("../data/test/test_dataset_v01.2.jsonl", "r").readlines()]
    request_output_path = "../data/requests/llms"
    os.makedirs(request_output_path, exist_ok=True)
    
    if args.w_profile:
        request_file_path = request_output_path+f"/{args.model}_w_profile"
    else:
        request_file_path = request_output_path+f"/{args.model}_wo_profile"
    
    with open(request_file_path+"_request.jsonl", "w") as write_file:
        get_request(write_file, test_dataset, tokenizer, args)
      
    client = OpenAI(api_key=open("../utils/openai_key", "r").read().strip())
    input_file = client.files.create(
        file=open(request_file_path+"_request.jsonl", "rb"),
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
            output_file = open(request_file_path+"_output.jsonl", "w")
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
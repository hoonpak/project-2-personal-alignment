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

def mk_claude_message(custom_id, model, user_prompt):
    return Request(
        custom_id=custom_id,
        params=MessageCreateParamsNonStreaming(
            model=model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": user_prompt
            }]
        )
    )

def get_request(requests, test_dataset, tokenizer, args):
    for sample in test_dataset:
        profile = sample['user_profile']
        query = tokenizer.decode(tokenizer.encode(sample['query'])[-384:]).strip()
        
        if args.w_profile:
            user_prompt = mk_llm_w_profile_prompt(profile, query)
        else:
            user_prompt = mk_llm_wo_profile_prompt(query)
            
        requests.append(mk_claude_message(sample['conv_id'], args.model, user_prompt))
        
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
        
    requests = []
    get_request(requests, test_dataset, tokenizer, args)
    
    client = anthropic.Anthropic(api_key=open("../utils/anthropic_key", "r").read().strip())
    
    message_batch = client.messages.batches.create(requests=requests)
    batch_id = message_batch.id
    i = 0
    st_time = time.time()
    while True:
        retrieved_batch = client.messages.batches.retrieve(batch_id)
        print(f"Batch {batch_id} processing status is {retrieved_batch.processing_status} (elapsed_time... {time.time() - st_time:.2f} sec)")
        if retrieved_batch.processing_status == 'ended':
            output_file = open(request_file_path+"_output.jsonl", "w")
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
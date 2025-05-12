import time
import re
import json
from openai import OpenAI
from argparse import ArgumentParser, BooleanOptionalAction
from transformers import AutoTokenizer

import sys
sys.path.append("../utils")

from prompt import *

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--confidence", action=BooleanOptionalAction)
    parser.add_argument("--chosen_file_path", type=str)
    parser.add_argument("--control_file_path", type=str)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k :<20}: {v}")
    test_dataset = [json.loads(line) for line in open("../data/test/test_dataset_v01.2.jsonl", "r").readlines()]
    chosens = [json.loads(line) for line in open(args.chosen_file_path, "r").readlines()]
    controls = [json.loads(line) for line in open(args.control_file_path, "r").readlines()]
    labels = open("../data/labels.txt", "r").read().split()
    
    chosen_name = "".join(args.chosen_file_path.split("/")[3:])
    control_name = "".join(args.control_file_path.split("/")[3:])

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

    with open(f"../data/requests/win_rate/confi_{str(args.confidence)}_{chosen_name}_{control_name}_request.jsonl", "w") as write_file:
        for idx in range(len(test_dataset)):
            query = tokenizer.decode(tokenizer.encode(test_dataset[idx]['query'])[-384:]).strip()
            profile = test_dataset[idx]['user_profile']
            chosen = chosens[idx]['response']
            control = controls[idx]['response']
            if labels[idx] == "A":
                A = chosen
                B = control
            else:
                B = chosen
                A = control
            win_rate_eval_prompt = mk_winrate_eval_prompt(user_profile=profile, query=query, A=A, B=B, confidence=args.confidence)
            custum_id = test_dataset[idx]['conv_id']
            seed = 42
            model = "gpt-4-1106-preview"
            tmp_request = mk_winrate_eval_request(custum_id, seed, model, win_rate_eval_prompt)
            write_file.write(json.dumps(tmp_request, ensure_ascii=False)+"\n")

    client = OpenAI(api_key=open("../utils/openai_key", "r").read().strip())
    input_file = client.files.create(
        file=open(f"../data/requests/win_rate/confi_{str(args.confidence)}_{chosen_name}_{control_name}_request.jsonl", "rb"),
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
            output_file = open(f"../data/requests/win_rate/confi_{str(args.confidence)}_{chosen_name}_{control_name}_output.jsonl", "w")
            for i in file_response.iter_lines():
                output_file.write(i + "\n")
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
    
    result =  [json.loads(line)["response"]["body"]["choices"][0]["message"]["content"] 
               for line in open(f"../data/requests/win_rate/confi_{str(args.confidence)}_{chosen_name}_{control_name}_output.jsonl", "r").readlines()]

    cor = 0
    cor_total = 0
    confi_cor_80 = 0
    confi_cor_80_total = 0
    confi_cor_60 = 0
    confi_cor_60_total = 0

    for pred, label in zip(result, labels):
        if (" ".join(pred.lower().split(" ")[:2]).lower() == 'output (a)') and (label == "A"):
            cor+=1
            if [int(num) for num in re.findall(r'\d+', pred.lower().split(" ")[-1])][0] > 80:
                confi_cor_80 += 1
            if [int(num) for num in re.findall(r'\d+', pred.lower().split(" ")[-1])][0] > 60:
                confi_cor_60 += 1
        if (" ".join(pred.lower().split(" ")[:2]).lower() == 'output (b)') and (label == "B"):
            cor+=1
            if [int(num) for num in re.findall(r'\d+', pred.lower().split(" ")[-1])][0] > 80:
                confi_cor_80 += 1
            if [int(num) for num in re.findall(r'\d+', pred.lower().split(" ")[-1])][0] > 60:
                confi_cor_60 += 1
        
        if (" ".join(pred.lower().split(" ")[:2]).lower() == 'output (a)') or (" ".join(pred.lower().split(" ")[:2]).lower() == 'output (b)'):
            cor_total += 1
        
        if ([int(num) for num in re.findall(r'\d+', pred.lower().split(" ")[-1])][0] > 80) and ((" ".join(pred.lower().split(" ")[:2]).lower() == 'output (a)') or (" ".join(pred.lower().split(" ")[:2]).lower() == 'output (b)')):
            confi_cor_80_total += 1
        if ([int(num) for num in re.findall(r'\d+', pred.lower().split(" ")[-1])][0] > 60) and ((" ".join(pred.lower().split(" ")[:2]).lower() == 'output (a)') or (" ".join(pred.lower().split(" ")[:2]).lower() == 'output (b)')):
            confi_cor_60_total += 1
            
    print("#" * 60)
    print(f"{'pair wise results:':<35}    {round((cor / 793)*100, 2):>10}")
    print(f"{'total evals:':<35}{cor:>10}/793")
    print(f"{'confidence pair 60 wise results:':<35}    {round((confi_cor_60 / confi_cor_60_total)*100, 2):>10}")
    print(f"{'confidence 60 total:':<35}{confi_cor_60:>10}/{confi_cor_60_total}")
    print(f"{'confidence pair 80 wise results:':<35}    {round((confi_cor_80 / confi_cor_80_total)*100, 2):>10}")
    print(f"{'confidence 80 total:':<35}{confi_cor_80:>10}/{confi_cor_80_total}")
    print("#" * 60)
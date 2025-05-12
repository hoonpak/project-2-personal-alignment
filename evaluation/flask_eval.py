import time
import json
from openai import OpenAI
from argparse import ArgumentParser
from transformers import AutoTokenizer

import os, sys
sys.path.append("../utils")

from prompt import *


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--response_path", type=str)
    args = parser.parse_args()
    
    test_datas = [json.loads(line) for line in open("../data/test/test_dataset_v01.2.jsonl", "r").readlines()]
    responses = [json.loads(line) for line in open(args.response_path, "r").readlines()]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

    score_rubric = f"""* Criteria 1: {personalized_rubric['criteria']}
* Score descriptions 1: {json.dumps(personalized_rubric['score_descriptions'], ensure_ascii=False, indent=4)}
* Criteria 2: {values_rubric['criteria']}
* Score descriptions 2: {json.dumps(values_rubric['score_descriptions'], ensure_ascii=False, indent=4)}
* Criteria 3: {diversity_rubric['criteria']}
* Score descriptions 3: {json.dumps(diversity_rubric['score_descriptions'], ensure_ascii=False, indent=4)}
* Criteria 4: {creativity_rubric['criteria']}
* Score descriptions 4: {json.dumps(creativity_rubric['score_descriptions'], ensure_ascii=False, indent=4)}
* Criteria 5: {fluency_rubric['criteria']}
* Score descriptions 5: {json.dumps(fluency_rubric['score_descriptions'], ensure_ascii=False, indent=4)}
* Criteria 6: {factuality_rubric['criteria']}
* Score descriptions 6: {json.dumps(factuality_rubric['score_descriptions'], ensure_ascii=False, indent=4)}
* Criteria 7: {helpfulness_rubric['criteria']}
* Score descriptions 7: {json.dumps(helpfulness_rubric['score_descriptions'], ensure_ascii=False, indent=4)}
* Criteria 8: {safety_rubric['criteria']}
* Score descriptions 8: {json.dumps(safety_rubric['score_descriptions'], ensure_ascii=False, indent=4)}"""

    with open(f"../data/requests/preference_wise/{"".join(args.response_path.split("/")[3:])}_request.jsonl", "w") as write_file:
        for idx in range(len(test_datas)):
            query = tokenizer.decode(tokenizer.encode(test_datas[idx]['query'])[-384:]).strip()
            profile = test_datas[idx]['user_profile']
            response = responses[idx]['response']
            sys_prompt, flask_eval_prompt = mk_flask_eval_prompt(query=query, user_profile=profile, response=response, score_rubric=score_rubric)
            custum_id = test_datas[idx]['conv_id']
            seed = 42
            model = "gpt-4-1106-preview"
            tmp_request = mk_flask_eval_request(custum_id, seed, model, sys_prompt, flask_eval_prompt)
            write_file.write(json.dumps(tmp_request, ensure_ascii=False)+"\n")
    
    client = OpenAI(api_key=open("../utils/openai_key", "r").read().strip())
    input_file = client.files.create(
        file=open(f"../data/requests/preference_wise/{"".join(args.response_path.split("/")[3:])}_request.jsonl", "rb"),
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
            output_file = open(f"../data/requests/preference_wise/{"".join(args.response_path.split("/")[3:])}_output.jsonl", "w")
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
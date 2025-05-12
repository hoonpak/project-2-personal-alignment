# Preparing Environment
```
tar -I unzstd -xvf data.tar.zst
conda create -n p_align python=3.12.9
conda activate p_align
pip install transformers==4.49.0 peft==0.14.1.dev0 torch==2.6.0
```

# Dataset
### PRISM
* Paper: https://arxiv.org/abs/2404.16019
* Project page: https://hannahkirk.github.io/prism-alignment/
* hugging face: https://huggingface.co/datasets/HannahRoseKirk/prism-alignment

# How to learn?
```
cd ./script
CUDA_VISIBLE_DEVICES="0" python sft_training.py --data_ver v01.2 --model_name Qwen/Qwen2.5-3B --grad_accum_steps 32 --sft_lr 5e-6 --sft_v 0
CUDA_VISIBLE_DEVICES="0" python dpo_training.py --data_ver v01.2 --model_name Qwen/Qwen2.5-3B --grad_accum_steps 64 --dpo_lr 5e-6 --sft_v 0 --dpo_v 0
```

# How to evaluationn?
## 1. Make your openai key for judging.
```
cd ./utils
echo "your openai key" > openai_key
```
## 2. Run evaluation file
```
cd ./evaluation

python win_rate_eval.py \
    --confidence \
    --chosen_file_path "../data/eval_output/Qwen/Qwen2.5-3B/v01.2/dpo/0/gpt-4o-2024-08-06.jsonl" \
    --control_file_path "../data/eval_output/llms/w_gpt-4o-2024-08-06.jsonl"

python flask_eval.py --response_path "../data/eval_output/Qwen/Qwen2.5-3B/v01.2/dpo/0/gpt-4o-2024-08-06.jsonl"
```
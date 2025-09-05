from vllm import LLM, SamplingParams
import os
import json
import sys
from datetime import datetime
from tqdm import tqdm

sys.path.append('../data')
from kernel_dataset import KernelDataset

HOME_PATH = os.path.expanduser('~')
PROJECT_PATH = os.path.join(HOME_PATH, 'UniPar')

def build_message(example):
    kernel_name, from_api, from_code, to_api, _ = example
    messages = [
        {"role": "system", "content": "You are an HPC expert specializing in translating between parallel programming APIs."},
        {"role": "user", "content": f"Translate the following {kernel_name} kernel code from {from_api} to {to_api}. Code:\n{from_code}"},
    ]
    return messages


llm = LLM(
    model="/tmp/torchtune/llama3_1_8B/lora_single_device/epoch_0",
    load_format="safetensors",
    kv_cache_dtype="auto",
)
sampling_params = SamplingParams(max_tokens=16384, temperature=0.2)
dataset_dir = os.path.join(PROJECT_PATH, 'data/Datasets/HeCBench')
test_set = KernelDataset(dataset_dir, dataset_type='test')

for example in test_set:
    import pdb; pdb.set_trace()
    conversation = build_message(example)

    outputs = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)

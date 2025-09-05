from transformers import AutoTokenizer
import os
import glob
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import re


if __name__ == '__main__':
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Collect all sample paths
    kernels = []
    with open('../Datasets/HeCBench/dataset.jsonl', 'r') as f:
        for line in f:
            sample = json.loads(line)
            kernels.append(sample)

    stats_file = 'HeCBench_stats.jsonl'

    # with open(stats_file, 'w') as f:
    #   for kernel in tqdm(kernels):

    #         sample_metadata = {
    #             'kernel': kernel['kernel_name'],
    #             'parallel_api': kernel['kernel_api'],
    #         }

    #         full_code = '\n'.join(kernel['code'].values())

    #         tokens = tokenizer.encode(full_code)
    #         sample_metadata['word_count'] = len(full_code.split())
    #         sample_metadata['token_count'] = len(tokens)  
    #         sample_metadata['lines_count'] = len(full_code.splitlines())

    #         f.write(json.dumps(sample_metadata) + '\n')

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    token_counts = []

    with open(stats_file, 'r') as jsonl_file:
        for line in jsonl_file:
            sample_metadata = json.loads(line)
            
            token_counts.append(sample_metadata['token_count'])

    bins = [idx*10**3 for idx in range(20)]

    plt.hist(token_counts, bins=bins, edgecolor='black')

    plt.title('Distribution of Token Counts')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.grid()
    
    plt.subplot(1, 2, 2)
    token_counts = []

    with open(stats_file, 'r') as jsonl_file:
        for line in jsonl_file:
            sample_metadata = json.loads(line)

            token_counts.append(sample_metadata['lines_count'])

    bins = [idx*10**2 for idx in range(20)]

    plt.hist(token_counts, bins=bins, edgecolor='black')

    plt.title('Distribution of Code Lines')
    plt.xlabel('Number of Lines')
    plt.ylabel('Frequency')
    plt.grid()

    plt.savefig('stats.png')

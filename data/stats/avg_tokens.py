import json
import numpy as np

tokens_amount = []

with open('HeCBench_stats.jsonl', 'r') as f:
    for line in f:
        sample = json.loads(line.strip())

        tokens_amount.append(sample['token_count'])


tokens_amount = np.array(tokens_amount)
print(f'mean: {tokens_amount.mean()}')
print(f'std: {tokens_amount.std()}')
print(f'median: {np.median(tokens_amount)}')

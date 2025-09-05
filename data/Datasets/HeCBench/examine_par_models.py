import json


kernels = []
NUM_MODELS = 5


for datafile in ['test.jsonl']:

    with open(datafile, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())
            kernel_name = sample['kernel_name']
            kernels.append(kernel_name)
        

possible_prompt_kernels = set()

for kernel in kernels:
    if kernels.count(kernel) == NUM_MODELS:
        possible_prompt_kernels.add(kernel)

print(f'Amount of kernels {len(possible_prompt_kernels)}')
print(possible_prompt_kernels)

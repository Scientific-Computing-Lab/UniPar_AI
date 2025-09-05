import json


for datafile in ['train.jsonl', 'validation.jsonl', 'test.jsonl', 'prompt.jsonl']:
    samples = []
    kernel_names = set()
    api_counts = {}

    with open(datafile, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())
            kernel_name = sample['kernel_name']
            parallel_api = sample['parallel_api']
        
            if kernel_name:
                kernel_names.add(kernel_name)
        
            if parallel_api:
                if parallel_api not in api_counts:
                    api_counts[parallel_api] = 0
                
                api_counts[parallel_api] += 1

    print(f'split: {datafile}')
    print("Kernel Names:", ", ".join(sorted(kernel_names)))

    print("API Counts:")
    for api, count in api_counts.items():
        print(f"{api}: {count}")



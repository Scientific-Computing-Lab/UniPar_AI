import json

kernels_length = {}

with open('../dataset.jsonl', 'r') as f:
    for line in f:
        sample = json.loads(line)

        kernel_name = sample['kernel_name']
        files = sample['code']

        if len(files) > 1:
            continue
        print(kernel_name)
        code = list(files.values())[0]
        if kernel_name in kernels_length:
            kernels_length[kernel_name]['api'] += 1
            kernels_length[kernel_name]['length'] = max(kernels_length[kernel_name]['length'], 
                                                        len(code)) 
        else:
            kernels_length[kernel_name] = {'api': 1, 'length': len(code)}

with open('prompt_potential.jsonl', 'w') as f:
    for kernel, stats in kernels_length.items():
        if stats['api'] == 5 and stats['length'] < 6000:
            f.write(json.dumps({'kernel_name': kernel, 'max_length': stats['length']}) + '\n')

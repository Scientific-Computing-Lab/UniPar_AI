import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def dataset_split(kernels, prompts=['accuracy', 'gabor', 'permute']):
    prompt_set = []
    for kernel in kernels:
        if kernel['kernel_name'] in prompts:
            prompt_set.append({'kernel_name': kernel['kernel_name'], 'parallel_api': kernel['kernel_api'], 'code': kernel['code']})

    kernel_names = list(set([kernel['kernel_name'] for kernel in kernels if  kernel['kernel_name'] not in prompts]))

    train_data, test_data = train_test_split(kernel_names, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
    
    train_set = []
    val_set = []
    test_set = []
    
    for kernel in tqdm(kernels):
        
        split = test_set if kernel['kernel_name'] in test_data else val_set if kernel['kernel_name'] in val_data else train_set
        the_code_dict = {k: v[0] for k, v in kernel['PCC']['gpt-4o-mini'].items()} if 'PCC' in kernel else kernel['code']
        split.append({'kernel_name': kernel['kernel_name'], 'parallel_api': kernel['kernel_api'], 'code': the_code_dict})
    return train_set, val_set, test_set, prompt_set


if __name__=='__main__':
    kernels = []
    with open('/home/tomerbitan/unipar/UniPar/data/Datasets/HeCBench/dataset_pcc.jsonl', 'r') as f:
        for line in f:
            sample = json.loads(line)
            kernels.append(sample)
            # assert '//' in list(sample['PCC']['gpt-4o-mini'].values())[0][0], sample['kernel_name']


    train, val, test, prompt = dataset_split(kernels)

    for split, file_name in [(train, 'train_pcc.jsonl'), (val, 'validation_pcc.jsonl'), (test, 'test_pcc.jsonl'), (prompt, 'prompt.jsonl')]:
        with open(file_name, 'w') as f:
            for kernel in split:
                f.write(json.dumps(kernel) + '\n')
                
            
            

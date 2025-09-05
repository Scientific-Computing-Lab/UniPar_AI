import os
import json
from kernel_dataset import KernelDataset

HOME_PATH = os.path.expanduser('~')
PROJECT_PATH = os.path.join(HOME_PATH, 'SC-lab', 'UniPar')


def build_message(kernel_name, from_api, from_code, to_api, to_code):
    return [
                {"role": "system", "content": "You are an HPC expert specializing in translating between parallel programming APIs."},
                {"role": "user", "content": f"Translate the following {kernel_name} kernel code from {from_api} to {to_api}. Code:\n{from_code}"},
                {"role": "assistant", "content": f' Here is the translated code:\n{to_code}'}
            ]


if __name__=='__main__':
    dataset_dir = os.path.join(PROJECT_PATH, 'data/Datasets/HeCBench')

    for dataset_type in ['train', 'test']:
        dataset = KernelDataset(dataset_dir, dataset_type=dataset_type)

        with open(f'{dataset_type}.jsonl', 'w') as f:
            for d in dataset:
                messages = {'messages': build_message(d[0], d[1], d[2], d[3], d[4])}
                f.write(json.dumps(messages) + '\n')

import json
import os 
import re
import glob
import random

import sys
sys.path.append('../..')
from utils.remove_pragma import remove_omp

HOME_PATH = os.path.expanduser('~')

redundant_line_comments_c = re.compile("//.*")
redundant_multiline_comments_c = re.compile("/\*.*?\*/", re.MULTILINE | re.DOTALL)


def remove_comments(code):
    code = redundant_line_comments_c.sub("\n", code)
    code = redundant_multiline_comments_c.sub("\n", code)

    return code

def load_code(file_path):
    with open(file_path, 'r', errors='ignore') as file:
        return file.read()


def create_dataset(kernels):
    dataset_path = os.path.join(HOME_PATH, 'HeCBench/src')
    dataset = []

    for kernel in kernels:
        kernel_name, kernel_api = kernel['kernel_name'].rsplit('-', 1)
        filenames = kernel['files']

        sample = {'kernel_name': kernel_name, 'kernel_api': kernel_api, 'code': {}}

        for path in filenames: 
            try:
                code = load_code(os.path.join(dataset_path, kernel['kernel_name'], path))
                code = remove_comments(code)

                sample['code'][os.path.basename(path)] = code

            except Exception as _:
                continue

        dataset.append(sample)

        # create serial 
        if kernel_api == 'omp':
            omp_sample = {'kernel_name': kernel_name, 'kernel_api': 'serial', 'code': {}}

            for filename, code in sample['code'].items(): 
                try:
                    omp_sample['code'][filename] = remove_omp(code)

                except Exception as _:
                    continue

            dataset.append(omp_sample)

    return dataset

if __name__=='__main__':
    
    kernels = []
    with open('kernel_files_analysis/relevant_files.jsonl') as f:
        for line in f:
            kernels.append(json.loads(line.strip()))

    dataset = create_dataset(kernels)
   
    with open('dataset.jsonl', 'w') as f:
        for d in dataset:
            f.write(json.dumps(d) + '\n')
                
            
            

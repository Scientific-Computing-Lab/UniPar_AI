import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt


def get_relevant_files(kernel_name, kernel_path):
    irrelevant_files = ['reference', 'util']
    code_types = ['.c', '.cpp', '.cu', '.sycl', '.hipsycl', '.cc']

    kernel = {}
    kernel['kernel_name'] = kernel_name
    all_files = []

    for root, dirs, files in os.walk(kernel_path):
        for file in files:
            all_files.append(os.path.join(root, file))

    # filter files
    all_files = [file for file in all_files if os.path.splitext(file)[1] in code_types]
    all_files = [file for file in all_files if not any(irr in file for irr in irrelevant_files)]

    kernel_files = [file for file in all_files if 'kernel' in file] 
    non_kernel_files = [file for file in all_files if 'kernel' not in file] 

    if len(non_kernel_files) > 2 or len(kernel_files) > 2:
        return None
    
    kernel['files'] = non_kernel_files + kernel_files
    return kernel


HOME_PATH = os.path.expanduser('~')
base_path = os.path.join(HOME_PATH, "HeCBench/src")
kernels = {}


with open('relevant_files.jsonl', 'w') as f:
    for kernel_name in os.listdir(base_path):
        if kernel_name in ["scripts", "include"]:
            continue
        
        sample = get_relevant_files(kernel_name, os.path.join(base_path, kernel_name))
        if sample:
            f.write(json.dumps(sample) + '\n')

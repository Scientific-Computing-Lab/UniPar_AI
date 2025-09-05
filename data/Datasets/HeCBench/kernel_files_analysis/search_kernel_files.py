import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt


relevant_files = ['kernel', 'reference', 'util']


def find_files(kernel_name, kernel_path):
    code_types = ['.c', '.cpp', '.cu', '.sycl', '.hipsycl', '.cc']
    exists = False

    kernel = {}
    kernel['kernel_name'] = kernel_name
    types = {t:False for t in code_types}

    for relevant_file in relevant_files:
        kernel[relevant_file] = False

    for root, dirs, files in os.walk(kernel_path):
        for file in files:

            _, ext = os.path.splitext(file)

            if ext in code_types:
                types[ext] = True
                exists = True

            if ext in code_types and relevant_file in file:
                kernel[relevant_file] = True

    # if not any(types.values()) :
    #     print(types)

    if not exists:
        print(kernel_name)
    
    return kernel


HOME_PATH = os.path.expanduser('~')
base_path = os.path.join(HOME_PATH, "HeCBench/src")
kernels = {}


for kernel_name in os.listdir(base_path):
    if kernel_name in ["scripts", "include"]:
        continue
    
    kernels[kernel_name] = find_files(kernel_name, os.path.join(base_path, kernel_name))

import pdb; pdb.set_trace()

amount = defaultdict(int)

for kernel in kernels.values():
    for t in relevant_files:
        if kernel[t]:
            amount[t] += 1

res =  {'total': 1758-1166, 'kernel': 244, 'reference': 47, 'util': 121}

plt.bar(res.keys(), res.values(), edgecolor='black')
plt.grid()
plt.xlabel('file name')
plt.ylabel('frequency')

plt.savefig('filename.png')

import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt


def count_files(kernel_name, kernel_path):
    kernel = {}
    kernel['kernel_name'] = kernel_name

    for root, dirs, files in os.walk(kernel_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext:
                kernel[ext.lower()] = kernel[ext.lower()] + 1 if ext.lower() in kernel else 1
    
    return kernel


HOME_PATH = os.path.expanduser('~')
base_path = os.path.join(HOME_PATH, "HeCBench/src")
kernels = {}

total = 0

for kernel_name in os.listdir(base_path):
    if kernel_name in ["scripts", "include"]:
        continue
    total += 1
    kernels[kernel_name] = count_files(kernel_name, os.path.join(base_path, kernel_name))

print(total)

# with open('count_file_types.jsonl', 'w') as f:
#     for kernel in kernels.values():
#         f.write(json.dumps(kernel) + '\n')


code_types = ['.c', '.cpp', '.cu', '.sycl', '.hipsycl', '.cc']
header_types = ['.h', '.hpp', '.hipcl']


files_counter = defaultdict(int)
max_files = 0

with open('0.txt', 'w') as f:
    for kernel in kernels.values():
        
        amount = sum([kernel[ty] for ty in code_types if ty in kernel])
        if amount == 0:
            f.write(f"{kernel['kernel_name']},")

        max_files = amount if amount > max_files else max_files
        files_counter[amount] += 1

        
print(files_counter)


# X = range(1, max_files+1)[:20]
# plt.bar(X, [files_counter[x] for x in X], edgecolor='black')
# print([files_counter[x] for x in X])

# plt.xlabel('Amount of files')
# plt.ylabel('Kernel Frequency')

# plt.grid()
# plt.title('code only')
# plt.savefig('code_only.png')



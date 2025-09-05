import os

base_path = "~/HeCBench/src"
apis_count = {}
kernels = set()

for kernel in os.listdir(base_path):
    if kernel in ["scripts", "include"]:
        continue
    
    (kernel_name, api) = kernel.rsplit('-', 1)
    kernels.add(kernel_name)
    apis_count[api] = 1 if api not in apis_count else apis_count[api]+1

print(apis_count, len(kernels))

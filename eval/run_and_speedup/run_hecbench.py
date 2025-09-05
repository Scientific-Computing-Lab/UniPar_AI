# #!/bin/bash
# counter = 0
# for dir in *-omp/ ; do
#   ((counter++))
#   if [ -f "$dir/Makefile.aomp" ]; then
#     echo "=== Running in ${dir%/} $counter/322==="
#     start=$(date +%s.%N)
#     (cd "$dir" && make run -f Makefile.aomp -j DEVICE=cpu)
#     end=$(date +%s.%N)
#     elapsed=$(awk "BEGIN { printf \"%.3f\", $end - $start }")
#     echo "=== Finished in ${elapsed}s ==="
#   else
#     echo "Skipping $dir (no Makefile.aomp)"
#   fi
# done

import os
import subprocess
import time
import json

def eval_run(benchmark_path, to_api, timeout_time = 620):
    execution_times = "--nothing--"

    os.chdir(benchmark_path)
    try:
        res = subprocess.run(['make','clean'], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_time)
        start = time.time()
        if to_api == 'omp':
            res = subprocess.run(['make','run', '-f', 'Makefile.aomp', '-j' ,'DEVICE=cpu'], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_time)
        else:
            res = subprocess.run(['make','run', '-j'], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_time)
    except subprocess.CalledProcessError as e:
        res = e
        print(f"Error running {benchmark_path}: {res.stderr}")
    except subprocess.TimeoutExpired as e:
        res = e
        print(f"Timeout expired for {benchmark_path}")
        return 1, res.stdout, execution_times
    else:
        if res.returncode == 0:
            elapsed = time.time() - start
            execution_times = elapsed
    
    return res.returncode, res.stdout, execution_times

if __name__=='__main__':
    HOME_PATH = os.path.expanduser('~')
    # to_run = ['ace-omp', 'atomicIntrinsics-omp', 'convolution1D-omp', 'geodesic-omp', 'heat-omp', 'keogh-omp', 'meanshift-omp', 'michalewicz-omp', 'mixbench-omp', 'particlefilter-omp', 'pnpoly-omp', 'pool-omp', 'slu-omp', 'stddev-omp', 'su3-omp', 'vanGenuchten-omp', 'winograd-omp', 'wsm5-omp', 'xsbench-omp', 'zmddft-omp','stencil1d-omp',]
    to_run = ['ace-cuda', 'atomicIntrinsics-cuda', 'convolution1D-cuda', 'geodesic-cuda', 'heat-cuda', 'keogh-cuda', 'meanshift-cuda', 'michalewicz-cuda', 'mixbench-cuda', 'particlefilter-cuda', 'pnpoly-cuda', 'pool-cuda', 'stddev-cuda', 'stencil1d-cuda', 'su3-cuda', 'vanGenuchten-cuda', 'winograd-cuda', 'wsm5-cuda', 'zmddft-cuda',]
            #   ]#['stencil1d', 'keogh', 'xsbench', 'stddev', 'columnarSolver', 'convolution1D', 'atomicIntrinsics', 'pnpoly', 'winograd', 'michalewicz', 'zmddft', 'geodesic']#'crc64', 'su3', 'hotspot3D', 'slu', 'mixbench', 'pool', 'meanshift', 'vanGenuchten', 'wsm5', 'heat', 'particlefilter',
    all_times = {}
    for name in to_run:
        api_loc = os.path.join(HOME_PATH, f'/home/tomerbitan/unipar/HeCBench/src/{name}')#}-omp')
        print(f'{name}')
        try:
            the_dir = os.listdir(api_loc)
        except FileNotFoundError:
            print(f"Skipping {api_loc} - directory does not exist")
            continue
        results = []
        suc_cuda = 0
        total_cuda = 0
        suc_omp = 0
        total_omp = 0
        if name.split('-')[-1] == 'omp':
            to_api = 'omp'
        else:
            to_api = 'cuda'
        result = eval_run(api_loc, to_api)
        all_times[f"{name}"] = result[2]
        print(f"Results for {name}: retun code {result[0]} execution times: {result[2]}, output: {result[1]}")
        
        print()

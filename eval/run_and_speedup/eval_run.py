#run this in terminal before running script: export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


import os
import subprocess
import time
import json
import sys
import multiprocessing
import argparse

def process_kernel(params):
    """Process a single kernel benchmark."""
    kernel, benchmark_path, to_api = params
    code_path = os.path.join(benchmark_path, 'src')
    curr_dir_path = os.path.join(code_path, kernel)
    
    # Check if directory exists
    if not os.path.isdir(curr_dir_path):
        return 0, 1, kernel, None  # Not run, but counted, no time
    
    os.chdir(curr_dir_path)
    try:
        subprocess.run(['make', 'clean'], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Make clean failed for {kernel} with error: {e.stderr}")
    try:
        start = time.time()
        timeout_time = 620 #480 whe n verify  # seconds
        if to_api == 'omp':
            res = subprocess.run(['make','run', '-f', 'Makefile.aomp', '-j' ,'DEVICE=cpu'], #, 'VERIFY=yes'
                                check=True, text=True, stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, timeout=timeout_time)
        else:
            res = subprocess.run(['make','run', '-j'], check=True, text=True, #, 'VERIFY=yes'
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                timeout=timeout_time)
    except subprocess.CalledProcessError as e:
        print(f"\n**Error running {kernel}: {benchmark_path}, {e.stderr}\n")
        return 0, 1, kernel, None  # Not run, but counted, no time
    except subprocess.TimeoutExpired as e:
        print(f"\n**Timeout expired for {kernel}: {benchmark_path}\n")
        return 0, 1, kernel, None  # Not run, but counted, no time
    
    if res.returncode == 0:
        elapsed = time.time() - start
        print(f"\n**Execution out {kernel}: {res.stdout.strip()}\n")
        return 1, 1, kernel, elapsed  # Run successfully, counted, with time
    else:
        return 0, 1, kernel, None  # Not run, but counted, no time

def eval_run(benchmark_path, to_api):
    code_path = os.path.join(benchmark_path, 'src')
    
    # Make sure the path exists
    if not os.path.exists(code_path):
        print(f"Path does not exist: {code_path}")
        return 0, 0, {}
    
  
    try:
        kernels = [k for k in os.listdir(code_path) if os.path.isdir(os.path.join(code_path, k))]
    except FileNotFoundError:
        print(f"Directory not found: {code_path}")
        return 0, 0, {}
    
    # Prepare parameters for parallel processing
    kernel_params = [(kernel, benchmark_path, to_api) for kernel in kernels]
    
    # Process kernels in parallel
    num_pool = 1 if to_api == 'omp' else 4 # Use 1 process for omp, otherwise use all available
    with multiprocessing.Pool(processes=min(len(kernel_params), num_pool)) as pool:
        results = pool.map(process_kernel, kernel_params)
    
    # Aggregate results
    num_run = sum(r[0] for r in results)
    total = sum(r[1] for r in results)
    
    # Create execution times dictionary
    execution_times = {r[2]: r[3] for r in results if r[3] is not None}
    
    return num_run, total, execution_times

if __name__=='__main__':
    HOME_PATH = os.path.expanduser('~')
    parser = argparse.ArgumentParser(description="Run kernel benchmarks for a target API")
    parser.add_argument("--target", choices=["cuda", "omp"], required=True, help="Target API to use for benchmarks")

    parser.add_argument('--run_names', nargs='+', required=True, help='List of run names to evaluate')

    
    args = parser.parse_args()
    run_names_to_process = args.run_names
    target_api = args.target
    print("Selected target API:", target_api)
    for run_name in run_names_to_process:
        BASE = os.path.join(HOME_PATH, f'unipar/Datasets/eval/{run_name}')
        print(BASE)
        try:
            the_dir = os.listdir(BASE)
        except FileNotFoundError:
            print(f"Skipping {BASE} - directory does not exist")
            continue
        results = []
        suc_cuda = 0
        total_cuda = 0
        suc_omp = 0
        total_omp = 0
        all_times = {}
        for translation in the_dir:
            if len(translation.rsplit('-')) < 3:
                print("skipping: ", translation)
                continue
            from_api = translation.rsplit('-')[1]
            to_api = translation.rsplit('-')[2]
            if from_api == to_api:
                continue
            if (to_api not in [target_api]) or (from_api not in ['cuda','omp','serial']):
                # print("skipping: ", to_api)
                continue
            
            print("\n********Translate total for: ",translation,'********\n')
            result = eval_run(os.path.join(BASE, translation), to_api)
            all_times[f"{from_api}-{to_api}"] = result[2]
            results.append(result)
            print(from_api,"-",to_api,f" pass rate: {(result[0]/ result[1]):.3f}")#100*
            if to_api == 'cuda':
                suc_cuda += result[0]
                total_cuda += result[1]
            elif to_api == 'omp':
                suc_omp += result[0]
                total_omp += result[1]

        with open(os.path.join(BASE, 'execution_times.json'), 'w') as f:
            json.dump(all_times, f, indent=4)
        print()
        if total_cuda > 0:
            print(f"cuda pass rate: {(suc_cuda/ total_cuda):.2f}")
        if total_omp > 0:
            print(f"omp pass rate: {(suc_omp/ total_omp):.2f}")

        
        for r in results:
            print(f"=ROUND({r[0]}/{r[1]},3)", end="\t")
        print()
        sys.stdout.flush()
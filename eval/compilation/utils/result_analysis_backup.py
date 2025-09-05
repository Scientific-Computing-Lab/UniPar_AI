import json
import os
import subprocess
import argparse

HOME_PATH = os.path.expanduser('~')

all_run_res= {}

def eval_compile(benchmark_path, to_api, run_name):
    total, num_compile, other_counter = 0, 0, 0
    code_path = os.path.join(benchmark_path, 'src')

    # legal_kernals = list(set([name.split('_')[0] for name in ['addBiasResidualLayerNorm_cuda', 'addBiasResidualLayerNorm_hip', 'addBiasResidualLayerNorm_sycl', 'atomicIntrinsics_cuda', 'atomicIntrinsics_hip', 'atomicIntrinsics_omp', 'atomicIntrinsics_serial', 'blas-dot_cuda', 'blas-dot_hip', 'blas-dot_sycl', 'blas-gemmBatched_cuda', 'blas-gemmBatched_hip', 'blas-gemmBatched_sycl', 'clock_cuda', 'clock_hip', 'columnarSolver_cuda', 'columnarSolver_hip', 'columnarSolver_omp', 'columnarSolver_serial', 'columnarSolver_sycl', 'convolution1D_cuda', 'convolution1D_hip', 'convolution1D_omp', 'convolution1D_serial', 'convolution1D_sycl', 'daphne_cuda', 'daphne_hip', 'gelu_cuda', 'gelu_hip', 'gelu_sycl', 'geodesic_cuda', 'geodesic_hip', 'geodesic_omp', 'geodesic_serial', 'geodesic_sycl', 'heat_cuda', 'heat_hip', 'heat_omp', 'heat_serial', 'heat_sycl', 'hotspot3D_cuda', 'hotspot3D_hip', 'hotspot3D_omp', 'hotspot3D_serial', 'hotspot3D_sycl', 'jaccard_cuda', 'jaccard_hip', 'jaccard_sycl', 'keogh_cuda', 'keogh_hip', 'keogh_omp', 'keogh_serial', 'keogh_sycl', 'meanshift_cuda', 'meanshift_hip', 'meanshift_omp', 'meanshift_serial', 'meanshift_sycl', 'michalewicz_cuda', 'michalewicz_hip', 'michalewicz_omp', 'michalewicz_serial', 'michalewicz_sycl', 'mixbench_cuda', 'mixbench_hip', 'mixbench_omp', 'mixbench_serial', 'mixbench_sycl', 'particlefilter_cuda', 'particlefilter_hip', 'pcc_cuda', 'pcc_hip', 'pcc_sycl', 'pnpoly_cuda', 'pnpoly_hip', 'pnpoly_omp', 'pnpoly_serial', 'pnpoly_sycl', 'pool_cuda', 'pool_hip', 'pool_omp', 'pool_serial', 'pool_sycl', 'qem_cuda', 'qem_hip', 'qem_sycl', 'qkv_cuda', 'qkv_hip', 'qkv_sycl', 'reverse2D_cuda', 'reverse2D_hip', 'reverse2D_sycl', 'rowwiseMoments_cuda', 'rowwiseMoments_hip', 'rowwiseMoments_sycl', 'scan3_cuda', 'scan3_hip', 'scan3_sycl', 'slu_hip', 'slu_omp', 'slu_serial', 'spmv_cuda', 'spmv_hip', 'spmv_sycl', 'spnnz_cuda', 'spnnz_hip', 'sps2d_cuda', 'sps2d_hip', 'stddev_cuda', 'stddev_hip', 'stddev_omp', 'stddev_serial', 'stddev_sycl', 'stencil1d_cuda', 'stencil1d_hip', 'stencil1d_omp', 'stencil1d_serial', 'stencil1d_sycl', 'su3_cuda', 'su3_hip', 'su3_omp', 'su3_serial', 'su3_sycl', 'vanGenuchten_cuda', 'vanGenuchten_hip', 'vanGenuchten_omp', 'vanGenuchten_serial', 'vanGenuchten_sycl', 'winograd_cuda', 'winograd_hip', 'winograd_omp', 'winograd_serial', 'winograd_sycl', 'wsm5_cuda', 'wsm5_hip', 'wsm5_omp', 'wsm5_serial', 'wsm5_sycl', 'xsbench_hip', 'xsbench_omp', 'xsbench_serial', 'xsbench_sycl']]))
    # print(len(legal_kernals)
    for kernel in os.listdir(code_path):
        # if kernel.split('-')[0] not in legal_kernals:
        #     print(f"Skipping {kernel} - not in legal kernels list")
        #     continue
        curr_dir_path = os.path.join(code_path, kernel)

        os.chdir(curr_dir_path)
        # print(curr_dir_path)
        # res = subprocess.run(['make', 'clean'], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if code_path.split('/')[-2] == 'HeCBench-omp-cuda' and kernel == 'geodesic_cuda':
            pass
        try:
            subprocess.run(['make', 'clean'], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Make clean failed for {kernel} with error: {e.stderr}")
        try:
            if to_api == 'omp':
                res = subprocess.run(['make', '-f', 'Makefile.aomp','DEVICE=cpu', '-j'], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                res = subprocess.run(['make', '-j'], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            # print(e.stderr)
            # print("Compilation failed for kernel:", kernel)
            res = e

            # continue
        else:
            num_compile +=1
            # print("Compilation succeeded for kernel:", kernel)
        # # if len(os.listdir(curr_dir_path)) > num_files:
        # #     other_counter += 1
        if res.returncode != 0:
            other_counter += 1
            # print(f"Compilation failed for {kernel} with error: {res.stderr}")
        
        total += 1
        suc_fail = 'SUCCSESS' if res.returncode == 0 else 'FAILED'

        all_run_res[run_name][code_path.split('/')[-2]][kernel] = suc_fail
        print(f"Kernel {kernel} compilation status: {suc_fail}")

    return num_compile, total#, other_counter


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Evaluate compilation results for given run names.")
    parser.add_argument('--run_names', nargs='+', required=True, help='List of run names to evaluate')
    args = parser.parse_args()

    run_names_to_process = args.run_names
    

    for run_name in run_names_to_process:#['eval_shot_0_m_15000_t_0.2_p_0.9_33','eval_shot_3_m_15000_t_0.2_p_0.9_new','eval_shot_1_m_15000_t_0.2_p_0.9_new']:
        BASE = os.path.join(HOME_PATH, f'unipar/Datasets/eval/{run_name}')#gpt_shot_{shot}_m_15000_t_0.2_p_0.9_{ending}')##{gpt_str}{is_pcc_str}shot_{shot}_m_{int(mx_tok)}_t_{temp}_p_{tp}_new')#'unipar/Datasets/eval/gpt_shot_1_m_15000_t_0.2_p_0.9')#f'unipar/Datasets/eval/shot_{shot}_m_{int(mx_tok)}_t_{temp}_p_{tp}')#add gpt back#####{is_pcc}shot_{shot}_m_{mx_tok}_t_{temp}_p_{tp}
        print(BASE)
        run_name = BASE.split('/')[-1]
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
        all_run_res[run_name] = {}
        for translation in the_dir:
            if len(translation.rsplit('-')) < 3:
                # print("skipping: ", translation)
                continue
            
            from_api = translation.rsplit('-')[1]
            to_api = translation.rsplit('-')[2]
            if from_api == to_api:
                continue
            if (to_api not in ['omp','cuda']) or (from_api not in ['omp', 'cuda', 'serial']):# 
                # print("skipping: ", to_api)
                continue
            
            print("translate total for: ",translation)
            all_run_res[run_name][translation] = {}
            result =eval_compile(os.path.join(BASE, translation), to_api, run_name)
            results.append((translation, result))
            print(from_api,"-",to_api,f" pass rate: {(result[0]/ result[1]):.2f}")#100*
            if to_api == 'cuda':
                suc_cuda += result[0]
                total_cuda += result[1]
            elif to_api == 'omp':
                suc_omp += result[0]
                total_omp += result[1]
        
        
        
        print()
        print(f"cuda pass rate: {(suc_cuda/ total_cuda):.2f}")
        print(f"omp pass rate: {(suc_omp/ total_omp):.2f}")
        
        for r in results:
            print(f"=ROUND({r[1][0]}/{r[1][1]},3)", end="\t")
        print()
    
    # Iterate over every pair of run_names to find differences in kernel compilation statuses
    run_names = list(all_run_res.keys())
    for i in range(len(run_names)):
        for j in range(i + 1, len(run_names)):
            run_a = run_names[i]
            run_b = run_names[j]
            # Find common translation keys between the two runs
            common_translations = set(all_run_res[run_a].keys()) & set(all_run_res[run_b].keys())
            for translation in common_translations:
                # Find common kernel keys between the two translations
                common_kernels = set(all_run_res[run_a][translation].keys()) & set(all_run_res[run_b][translation].keys())
                for kernel in common_kernels:
                    status_a = all_run_res[run_a][translation][kernel]
                    status_b = all_run_res[run_b][translation][kernel]
                    # Check if one succeeded and the other failed
                    if (status_a == "SUCCSESS" and status_b == "FAILED") or \
                       (status_a == "FAILED" and status_b == "SUCCSESS"):
                        print(f"Run '{run_a}' and Run '{run_b}', Translation '{translation}', Kernel '{kernel}': {status_a} vs {status_b}")
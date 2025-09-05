#file name needs to be full path with home
file_list = ['/home/tomerbitan/unipar/UniPar/eval/run_and_speedup/resan_final_gpt_neval_cuda.txt', ]
all_kernal_names = []
for filename in file_list:
    filename ='/home/tomerbitan/unipar/UniPar/eval/run_and_speedup/res_anal_run_with_replace_omp_2.txt'#'/home/tomerbitan/unipar/UniPar/eval/run_and_speedup/res_anal_run_with_replace_omp_2_new.txt'#/home/tomerbitan/unipar/UniPar/eval/run_and_speedup/res_anal_run_with_replace_omp_2.txt'
    kernal_idicator = '**Execution out' if 'replace' in filename else 'Execution out'
    pair_idicator = '********Translate total for:' if 'replace' in filename else 'translate total for:'

    with open(filename, 'r') as file:
        content = file.read()

    configs = content.split('=ROUND')
    for config in configs:
        if not kernal_idicator in config:
            continue
        config_name = config.split('\n')[0] if '/home' in config.split('\n')[0] else config.split('\n')[1]#get path name catch the /home of path
        print(f"Processing config: {config_name}")
        api_pairs = config.split(pair_idicator)
        fail_num, pass_num, total_num = 0, 0 , 0

        for pair in api_pairs:
            if not kernal_idicator in pair:
                continue
            prair_name = pair.split('\n')[0]
            print(prair_name.strip('*').strip().strip('HeCBench-'))
            kernals_out = pair.split(kernal_idicator)
            passes = [kernal_out.split(':')[0] for kernal_out in kernals_out if 'PASS' in kernal_out]
            fails = [kernal_out.split(':')[0] for kernal_out in kernals_out if 'FAIL' in kernal_out]
            kernals_out_names = [kernal_out.split(':')[0] for kernal_out in kernals_out if ':' in kernal_out]
            pass_num += len(passes)
            fail_num += len(fails)
            total_num += len(kernals_out)-1
            print(f"Found {len(passes)} passes and {len(fails)} fails out of {len(kernals_out)-1}\n{passes}")
            all_kernal_names += kernals_out_names

        print(f"total passes: {pass_num}, total fails: {fail_num}, total: {total_num}, pass rate: {pass_num/total_num:.3f}")

    print(f"Total unique kernal names: {set(all_kernal_names)}")
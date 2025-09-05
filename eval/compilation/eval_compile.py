import json
from pathlib import Path
import os
import re
from tqdm import tqdm
import shutil
import itertools
import argparse


################################################################
# This evaluation will ignore kernels with more than one file! #
################################################################
# test 458 / 544

################################################################
# This evaluation will ignore kernels with more than one file! #
################################################################
# test 458 / 544

HOME_PATH = os.path.expanduser('~')
PROJECT_PATH = os.path.join(HOME_PATH, 'unipar', 'UniPar')

files_mapping = []
files_mapping_path = 'unipar/UniPar/data/Datasets/HeCBench/kernel_files_analysis/relevant_files.jsonl'
with open(os.path.join(HOME_PATH, files_mapping_path), 'r') as f:
    for line in f:
        files_mapping.append(json.loads(line))

files_mapping = {mapping["kernel_name"]: mapping["files"] for mapping in files_mapping}


def get_kernel_files(kernel_name):
    kernel, api1, _ = kernel_name.rsplit('_', 2)

    api1 = 'omp' if api1 == 'serial' else api1
    return set([os.path.basename(file) for file in files_mapping[f'{kernel}-{api1}']])


def extract_code_blocks(kernel_name, text):
    """
    The generated code maybe either in a code-block or in the format of the prompt.
    """
    def take_from_include(text):
        """
        Extract the code from the first include line.
        """
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("#include"):
                return "\n".join(lines[i:]).strip()
        return text.strip()
    code_blocks = re.findall(r"```(.*?)```", text, re.DOTALL)#remove ```???
    if code_blocks: 
        code = code_blocks[0]

        if code.startswith("cpp"):
            code = code[3:]
        elif code.startswith("c"):
            code = code[1:]
                
        code = re.sub(r'^(?:.*\.(c|cpp|cu|hip):|main:)\s*', '', code.strip())#r'^.*\.(c|cpp|cu|hip):\s*'
        code = take_from_include(code)
        if all([line not in code for line in ['#include','int ','__global__ ', '#define' ]]):
            text[list(re.finditer(r'```', text))[1].end():]
        return code
    else:
        delim = list(get_kernel_files(kernel_name))[0]
        matches = list(re.finditer(f'{os.path.splitext(delim)[0]}.*?:', text))

        if len(matches) == 1:
            return text[matches[0].end()+1:]
        elif len(matches) > 1:
            return take_from_include(text)

    # print(kernel_name)
    return "- Incompilable Code -"#maybe change this
    # match = re.search(r"```(?:\w+\n)?(.*?)```", text, re.DOTALL)
    # if match:
    #     return match.group(1).strip()
    # # Fallback: try to find the first #include and return from there
    # lines = text.splitlines()
    # for i, line in enumerate(lines):
    #     if line.strip().startswith("#include"):
    #         return "\n".join(lines[i:]).strip()
    # return text.strip()


def copy_structure_with_symlinks(source_dir, destination_dir, generated_kernels):
    source_dir = Path(source_dir).resolve()
    destination_dir = Path(destination_dir).resolve()
    changed_makefile_dirs = []  # List to track directories with changed Makefiles
    
    if not source_dir.exists():
        return changed_makefile_dirs

    for root, _, files in tqdm(os.walk(source_dir)):
        root_path = Path(root)

        if 'heat' in str(root_path):
            pass

        if '.git' in root:
            continue

        relative_root = root_path.relative_to(source_dir)
        if relative_root.parts and relative_root.parts[0] == "src":
            if not any(kernel in relative_root.parts for kernel in generated_kernels):
                continue

        relative_root = Path(root).relative_to(source_dir)
        target_root = destination_dir / relative_root
        target_root.mkdir(parents=True, exist_ok=True)

        for file in files:
            if '.git' in file:
                continue

            source_file = Path(root) / file
            target_file = target_root / file

            if target_file.exists() or target_file.is_symlink():
                target_file.unlink()
            
            if file.lower().split('.')[0] == 'makefile':
                shutil.copy2(source_file, target_file)
                with open(target_file, 'r') as f:
                    content = f.read()
                original_content = content
                modified_content = re.sub(
                    r'\.\./([^\s:="\']+)', 
                    lambda m: str(source_dir / 'src' / m.group(1)),
                    content
                )

                # Check if the content was actually modified
                if modified_content != original_content:
                    # Add parent directory name to the list of changed dirs
                    parent_dir = str(target_root.name)
                    changed_makefile_dirs.append(parent_dir)
                    
                with open(target_file, 'w') as f:
                    f.write(modified_content)
            else:
                target_file.symlink_to(source_file)
    
    return changed_makefile_dirs

    return changed_makefile_dirs

def update_hecbench(from_api, to_api, gen_code_path, hecbench_path, temp_dir):
    generated_kernels = os.listdir(gen_code_path)
    total = len(generated_kernels)
    generated_kernels = list(filter(lambda kernel: len(get_kernel_files(kernel))==1, generated_kernels))
    print(f'Amount of kernels to be analyzed: {len(generated_kernels)}/{total}')

    generated_kernels_cp = [f"{kernel.rsplit('_', 2)[0]}-{kernel.rsplit('_', 2)[2]}"
                          for kernel in generated_kernels if kernel.rsplit('_', 2)[1] == from_api and kernel.rsplit('_', 2)[2] == to_api]

    changed_dirs = copy_structure_with_symlinks(hecbench_path, temp_dir, generated_kernels_cp)
    
    # Print directories with changed Makefiles if any
    if changed_dirs:
        print(f"Modified Makefiles in the following directories: {', '.join(changed_dirs)}")
    
    # edit main files
    dataset_dir = os.path.join(PROJECT_PATH, 'data/Datasets/HeCBench')
    with open(os.path.join(dataset_dir, 'dataset.jsonl'), 'r') as f:
        kernels = [json.loads(line) for line in f]
    to_codes = {f"{kernel['kernel_name']}-{kernel['kernel_api']}": '\n'.join(f'{filename}:\n{code}' for filename, code in kernel['code'].items()) for kernel in kernels}
    for gen_path in tqdm(generated_kernels):
        kernel_name, from_kernel, to_kernel = gen_path.rsplit('_', 2)
        api = 'omp' if to_kernel=='serial' else to_kernel
        path = files_mapping[f'{kernel_name}-{api}'][0]
        
        pred_name = 'pred_0.cpp'
        if not os.path.exists(os.path.join(gen_code_path, gen_path, pred_name)):
            pred_name = 'pred.txt'
        if not os.path.exists(os.path.join(gen_code_path, gen_path, 'truth.cpp')):
                with open(os.path.join(gen_code_path, gen_path, 'truth.cpp'), 'w') as f:
                    f.write(to_codes[f"{kernel_name}-{api}"])
        with open(os.path.join(gen_code_path, gen_path, pred_name), 'r') as f, \
             open(os.path.join(gen_code_path, gen_path, 'truth.cpp'), 'r') as f_gt:
            gt_code = f_gt.read()

            gt_includues = '\n'.join([line for line in gt_code.split("\n") if line.lstrip().startswith("#include") and "reference" not in line])
            gen_code = extract_code_blocks(gen_path, f.read())

        if from_kernel == from_api and to_kernel == to_api:
            # if os.path.exists(path.replace('HeCBench', temp_dir)):
            os.remove(path.replace('HeCBench', temp_dir))
            # os.makedirs(os.path.dirname(path.replace('HeCBench', temp_dir)), exist_ok=True)
            with open(path.replace('HeCBench', temp_dir), 'w') as f:
                f.write(gt_includues + '\n' + gen_code)


if __name__ == '__main__':
    from_apis = ['serial', 'omp', 'cuda', 'hip', 'sycl']
    to_apis = ['omp', 'cuda', 'hip', 'sycl']

    parser = argparse.ArgumentParser(description="Evaluate compilation results for given run names.")
    parser.add_argument('--model_out_files', nargs='+', required=True, help='List of run names to evaluate')
    parser.add_argument('--extracted_code_paths', nargs='+', required=True, help='List of extracted code paths')
    args = parser.parse_args()

    run_names_to_process = args.model_out_files
    extracted_code_paths = args.extracted_code_paths
    

    for res_name, extracted_code_path in zip(run_names_to_process, extracted_code_paths):
        for from_api, to_api in itertools.product(from_apis, to_apis):
            if from_api != to_api:
                gen_code_dir = os.path.join(HOME_PATH, f'unipar/Datasets/{extracted_code_path}')

                if not os.path.exists(gen_code_dir):
                    print(f"Skipping {gen_code_dir} - directory does not exist")
                    continue
                print(f"Processing {gen_code_dir}")
                update_hecbench(from_api=from_api,
                                to_api=to_api,
                                gen_code_path=os.path.join(HOME_PATH, gen_code_dir),
                                hecbench_path=os.path.join(HOME_PATH, 'unipar/HeCBench'),
                                temp_dir=os.path.join(HOME_PATH, f'unipar/Datasets/eval/{res_name}/HeCBench-{from_api}-{to_api}'))
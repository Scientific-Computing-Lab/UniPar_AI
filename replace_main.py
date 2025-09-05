import os
import re
import shutil
import subprocess
import multiprocessing
import time
import random
from openai import RateLimitError

def take_from_include(text):
        """
        Extract the code from the first include line.
        """
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("#include"):
                return "\n".join(lines[i:]).strip()
        return text.strip()

def gpt_fix(new_code, error_msg, original_main_func):
    from openai import AzureOpenAI

    endpoint = ''
    model_name = "gpt-4o-mini"
    deployment = "gpt-4o-mini"#-2"

    subscription_key = ''
    api_version = "2025-01-01-preview"

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    messages = [
        {"role": "system", "content": "You are an HPC expert specializing in translating between parallel programming APIs. you replay with code only starting with the necessary includes and no text after"},
        {"role": "user", "content": f"When insreting the main function into this code {new_code}. This is the error is recived on compilation \n{error_msg}. Correct the code. here is the original main function{original_main_func}\n\nProvide the new corrected code complete code. Do not truncate or use ellipses. Ensure correctness. All function names must match.\nNO TEXT EXCEPT THE CODE"},
    ]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            outputs_batch = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=messages,
                max_tokens=int(16000),
                temperature=0.2,
                top_p=0.9,
                n=1
            )
            return take_from_include(outputs_batch.choices[0].message.content)
        except RateLimitError as e:
            if attempt < max_retries - 1:
                delay = 15 ** attempt + random.uniform(0, 1)
                print(f"Rate limit error: {str(e)}. Retrying in {delay:.1f}s (Attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"Max retries reached after error: {str(e)}")
        except Exception as e:
            print("error when trying to fix the code with gpt: "    + str(e))

    
    return new_code

def find_main_functions(file_path):
    """Check if a file has 'int main' or 'void main' functions."""
    try:
        with open(file_path, 'r', errors='ignore') as file:
            content = file.read()
            pattern = r'(int|void)\s+main\s*\('
            return bool(re.search(pattern, content))
    except Exception as e:
        # print(f"Error reading {file_path}: {e}")
        return False

def get_main_func(content):
        main_pattern = r'(int|void)\s+main\s*\('#[^{]*\{[^{}]*(?:{[^{}]*}[^{}]*)*\}'
        loc_start_main = re.search(main_pattern, content, re.DOTALL)
        remain = content[loc_start_main.end():]
        all_open = [it.start() for it in re.finditer('{', remain)]
        all_close = [it.start() for it in re.finditer('}', remain)]
        merged_back_list = [(index, '{' ) for index in all_open] + [(index, '}' ) for index in all_close]
        merged_back_list.sort(key=lambda x: x[0])
        assert merged_back_list[0][1] == '{', "The first character should be an opening brace"
        count = 0
        end_index = None#len(content)
        for index, char in merged_back_list:
            if char == '{':
                count += 1
            else:
                count -= 1
            if count == 0:
                end_index = loc_start_main.end() + index + 1
                break
        if end_index == None:#len(content):
            print("No matching closing brace found for main function.")
            end_index = len(content)
        return content[loc_start_main.start():end_index], loc_start_main.start(), end_index
def compile(dir_path):
    os.chdir(dir_path)
    res = subprocess.run(['make', 'clean'], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        to_api = dir_path.split('-')[-1]
        if to_api == 'omp':
            res = subprocess.run(['make', '-f', 'Makefile.aomp','DEVICE=cpu', '-j'], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            res = subprocess.run(['make', '-j'], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        res = e
        return res.returncode == 0, res.stderr
    return res.returncode == 0, "Compilation succeeded"
def replace_with_original(src_file, dest_file, fix_attempt=True):
    """Replace only the main function in the destination file with the main function from the source file."""
    try:
        # Read source file to extract main function
        with open(src_file, 'r', errors='ignore') as file:
            src_content = file.read()
        
        # Read destination file
        with open(dest_file, 'r', errors='ignore') as file:
            dest_content = file.read()
        
        # Extract main function from source file
        src_main, _, _ = get_main_func(src_content)
        if not src_main:
            print(f"Could not find main function in source file: {src_file}")
            return False
            
        # Replace main function in destination content
        orig_dest_main, dest_main_start, dest_main_end = get_main_func(dest_content)
        dest_updated = dest_content[:dest_main_start] + src_main + dest_content[dest_main_end:]
        
        # Write updated content back to destination file
        # Create backup of destination file
        backup_file = dest_file + '.bak'
        shutil.copy2(dest_file, backup_file)
        curr_dir_path = os.path.dirname(dest_file)
        comp_before, _ = compile(curr_dir_path)
        with open(dest_file, 'w') as file:
            file.write(dest_updated)
        if 'keogh-cuda' in dest_file:
            pass
        comp_after, error_msg = compile(curr_dir_path)
        if (comp_before and not comp_after) and fix_attempt:
            print(f"Compilation failed after replacing main function in {dest_file}\n Error: {error_msg}\n backaup: {backup_file}.")
            fixed_code = gpt_fix(dest_updated, error_msg, orig_dest_main)
            with open(dest_file, 'w') as file:
                file.write(fixed_code)
            print("gpt fixed the code")
            comp_after_after, error_msg_after = compile(curr_dir_path)
            if not comp_after_after:
                shutil.copy2(backup_file, dest_file)
                os.remove(backup_file)
            return True if comp_after_after else False

            
        print(f"Replaced main function in {dest_file} with main function from {src_file}")
        return True
    except Exception as e:
        print(f"Error replacing main function from {src_file} to {dest_file}: {e}")
        return False

def process_file(params):
    """Process a single file to replace main function."""
    file_path, original_file = params
    
    if find_main_functions(file_path):
        if os.path.exists(original_file):
            if replace_with_original(original_file, file_path):
                return 1  # File was replaced
            return 0  # Failed to replace
        else:
            print(f"Original file not found: {original_file}")
            return 0
    return 0

def process_shot(params):
    """Process a single shot with the given parameters."""
    is_gpt, shot, mx_tok, temp, tp, ignore_sets = params
    
    if (shot, mx_tok, temp, tp) in ignore_sets:
        return 0, 0  # No files checked or replaced
    
    target_dir = f'/home/tomerbitan/unipar/Datasets/eval/FT_llama70_gen_omp_cuda_only_replace_new'
    hecbench_dir = '/home/tomerbitan/unipar/HeCBench/src/'
    
    print(f"Processing {is_gpt} shot {shot} with parameters: max_tokens={mx_tok}, temp={temp}, top_p={tp}")
    
    files_to_process = []
    
    # First, collect all the files that need processing
    for root, api_to_dirs, files in os.walk(target_dir):
        # if any(file.endswith('.bak') for file in files):
        #     continue
        if not files or ('Makefile' not in files):
            continue
        for file in files:
            if 'Makefile' in file or 'LICENSE' == file:
                continue
            file_path = os.path.join(root, file)
            original_file = os.path.join(hecbench_dir, root.split('/')[-1], file)
            files_to_process.append((file_path, original_file))
    
    
    # Process files in parallel
    with multiprocessing.Pool(processes=15) as pool:
        results = pool.map(process_file, files_to_process)
    
    files_checked = len(files_to_process)
    files_replaced = sum(results)
    
    print(f"\n{is_gpt} Shot {shot} Summary: Checked {files_checked} files, replaced {files_replaced} files with main functions.")
    return files_checked, files_replaced

def main():    
    # Hardcoded paths
    ignore_sets = []
    gpt_options = ['gpt_']  #'eval_',  'gpt_' ,
    
    total_files_checked = 0
    total_files_replaced = 0
    
    for is_gpt in gpt_options:
        for mx_tok in [15000]:  # , 10000,  5000]:#
            for temp in [0.2]:  # , 0.9]:#6,
                for tp in [0.9]:  # , 0.8]:
                    # Prepare parameters for each shot
                    shot_params = [(is_gpt, shot, mx_tok, temp, tp, ignore_sets) 
                                  for shot in [1, 2, 3]]  # 0,
                    for shot in [0]:
                        process_shot((is_gpt, shot, mx_tok, temp, tp, ignore_sets))
                    
                    

if __name__ == "__main__":
    main()
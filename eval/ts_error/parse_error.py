import os
import glob
import os
import re
import json
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser
import glob

HOME_PATH = os.path.expanduser('~/unipar')#os.path.expanduser('~')

CPP_LANGUAGE = Language(tscpp.language())
parser = Parser(CPP_LANGUAGE)

files_mapping = []
files_mapping_path = '/home/tomerbitan/unipar/UniPar/data/Datasets/HeCBench/kernel_files_analysis/relevant_files.jsonl'

with open(os.path.join(HOME_PATH, files_mapping_path), 'r') as f:
    for line in f:
        files_mapping.append(json.loads(line))

files_mapping = {mapping["kernel_name"]: mapping["files"] for mapping in files_mapping}


def is_error_node(node):
    if node.type == "ERROR":
        return True

    for child in node.children:
        if is_error_node(child):
            return True

    return False


def get_kernel_files(kernel_name):
    kernel, api1, _ = kernel_name.rsplit('_', 2)

    api1 = 'omp' if api1 == 'serial' else api1
    return set([os.path.basename(file) for file in files_mapping[f'{kernel}-{api1}']])


def extract_code_blocks(kernel_name, text):
    """
    The generated code maybe either in a code-block or in the format of the prompt.
    """
    code_blocks = re.findall(r'```[a-zA-Z]*\n(.*?)```', text, re.DOTALL)

    code_blocks = [re.sub(r'^(?:.*\.(c|cpp|cu):|main:)\s*', '', code.strip()) for code in code_blocks]

    if code_blocks: 
        return code_blocks
    else:
        file_delimiters = [f"{filename}:" for filename in get_kernel_files(kernel_name)]

        if all(text.count(delim) == 1 for delim in file_delimiters):
            indxs = [(text.find(delim), len(delim)) for delim in file_delimiters]
            sorted_indxs = sorted(indxs, key=lambda x: x[0]) + [(len(text), 0)]

            code_blocks = []

            for itr in range(0, len(sorted_indxs) - 1):
                delim_location, delim_length = sorted_indxs[itr]
                next_delim_location, _ = sorted_indxs[itr+1]

                code_blocks.append(text[delim_location + delim_length : next_delim_location])

            return code_blocks

    return []


def make_tree(kernel_name, file_name):

    with open(file_name, 'r') as f: 
        codes = extract_code_blocks(kernel_name, f.read())

    if len(codes) == 0:
        return None

    tree = parser.parse('\n'.join(codes).encode())
    return tree.root_node


def calc_ts_error(path):
    total, count = 0, 0
    # print(path)
    for kernel in os.listdir(path):
        # print(kernel)
        kernel_path = os.path.join(path, kernel)

        if not os.path.isdir(kernel_path):
            continue

        pred_files = glob.glob(os.path.join(kernel_path, "pred*"))
        total += 1 

        trees = [make_tree(kernel, f) for f in pred_files]
        trees = [t for t in trees if t is not None]

        # if any(is_error_node(t) for t in trees):
        if len(trees)==0 or is_error_node(trees[0]):
            count += 1 

    return total, count


if __name__=='__main__':
    path = os.path.join(HOME_PATH, "Datasets")

    for d in glob.glob(os.path.join(path, 'vllm_gpt-4o-mini_pcc_shots=0_max_token=15000.0_temp=0.2_p=0.9')):#'vllm_gpt-4o-mini_*')):#vllm_llama3_70b_')):#
        print(d)

        total, count = calc_ts_error(os.path.join(path, d))
        print(f"error node rate: {count}/{total}, {count/total:.3f}")



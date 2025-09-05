import os
import glob
import os
import re
import json
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser
from tqdm import tqdm

HOME_PATH = os.path.expanduser('~')

CPP_LANGUAGE = Language(tscpp.language())
parser = Parser(CPP_LANGUAGE)


redundant_line_comments_c = re.compile("\/\/.*")
redundant_multiline_comments_c = re.compile("\/\*.*?\*\/", re.MULTILINE|re.DOTALL)


def get_functions(code):
    tree = parser.parse(code.encode())
    root_node = tree.root_node

    functions = []
    def extract_function_names(node):
        if node.type == "function_definition":
            for child in node.children:
                if child.type == "function_declarator":
                    function_name = None
                    for subchild in child.children:
                        if subchild.type == "identifier":
                            function_name = subchild.text.decode()
                    if function_name:
                        functions.append(function_name)

        for child in node.children:
            extract_function_names(child)
    
    extract_function_names(root_node)
    return functions


def remove_comments(code):
    code = redundant_line_comments_c.sub("\n", code)
    code = redundant_multiline_comments_c.sub("\n", code)

    return code



def same_functions(pred_f, gt_f):

    with open(pred_f, 'r') as pf, open(gt_f, 'r') as gtf: 
        pred_code = extract_code_blocks(pf.read())
        if len(pred_code) == 0:
            return False
        pred_code = pred_code[0]
        gt_code = gtf.read()

        pred_funcs = get_functions(pred_code)
        gt_funcs = get_functions(gt_code)

    return any(func not in pred_funcs for func in gt_funcs)


def extract_code_blocks(text):
    code_blocks = re.findall(r'```[a-zA-Z]*\n(.*?)```', text, re.DOTALL)
    return code_blocks


def analyze_functions_in_directory(path):
    total, count = 0, 0
   
    for kernel in os.listdir(path):
        kernel_path = os.path.join(path, kernel)

        if not os.path.isdir(kernel_path):
            continue

        pred_files = glob.glob(os.path.join(kernel_path, "pred*"))
        truth_file = os.path.join(kernel_path, "truth.cpp")

        total += 1 


        if any(same_functions(f, truth_file) for f in pred_files):
            count += 1 

    return total, count

path = os.path.join(HOME_PATH, "atasets/temp_p")
for d in os.listdir(path):
    
    if '_p=0.9' in d:
        print(d)

        total, count = analyze_functions_in_directory(os.path.join(path, d))
        print(f"Total directories with truth.cpp: {total}")
        print(f"Directories with matching functions: {count}")


# vllm_vllm_llama3_8b_eval_shots=0_temp=0.2_p=0.9
# Total directories with truth.cpp: 395
# Directories with matching functions: 163
# vllm_vllm_llama3_8b_eval_shots=0_temp=0.7_p=0.9
# Total directories with truth.cpp: 400
# Directories with matching functions: 234
# vllm_vllm_llama3_8b_eval_shots=0_temp=1.0_p=0.9
# Total directories with truth.cpp: 399
# Directories with matching functions: 310


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

def print_tree_as_str(node, level=0):
    """
    Recursively print the entire tree starting from the given node as a string.

    Args:
        node: The starting node of the tree (e.g., Tree-Sitter Node).
        level: The current depth of the tree (used for indentation).
    """
    # Create indentation based on the level of the node
    indent = "  " * level
    # Start with the node's type
    node_str = f"{indent}{node.type}"

    # If the node has a text range (optional), include it
    if hasattr(node, 'start_byte') and hasattr(node, 'end_byte'):
        node_str += f" [{node.start_byte}:{node.end_byte}]"

    # Print the current node
    print(node_str)

    # Recursively process each child
    for child in node.children:
        print_tree_as_str(child, level + 1)


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


def iter_files(root_dir):

    for kernel in tqdm(os.listdir(root_dir)):
        file_path = os.path.join(root_dir, kernel)
        
        try:
            with open(os.path.join(file_path, 'pred_0.cpp'), 'r', encoding='utf-8') as pred_f, \
                 open(os.path.join(file_path, 'truth.cpp'), 'r', encoding='utf-8') as truth_f:
                
                code = remove_comments(pred_f.read())
                pred_functions = get_functions(code)
                
                code = remove_comments(truth_f.read())
                truth_functions = get_functions(code)

            with open(os.path.join(HOME_PATH, 'UniPar/data/stats/functions.jsonl'), 'a') as f:
                f.write(json.dumps({kernel: {'pred': pred_functions, 'truth': truth_functions}}) + '\n')

        except Exception as e:
            print(f"Error reading file {file_path}: {e}")


def analyze_functions(file_path):
    total = 0
    omit_func, new_func = 0, 0

    with open(file_path, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())

            pred_functions = list(sample.values())[0]['pred']
            truth_functions = list(sample.values())[0]['truth']

            new_func += any(func not in truth_functions for func in pred_functions)
            omit_func += any(func not in pred_functions for func in truth_functions)

            # if any(func not in truth_functions for func in pred_functions):
            #     import pdb; pdb.set_trace()
            #     asd = set(pred_functions)
            #     bsd = set(truth_functions)
            #     print(asd-bsd)

            # if any(func not in pred_functions for func in truth_functions):
            #     import pdb; pdb.set_trace()
            #     a = set(pred_functions)
            #     b = set(truth_functions)
            #     print(b-a)

            total += 1

    return total, new_func, omit_func


if __name__ == "__main__":
    root_directory = os.path.join(HOME_PATH, 'UniPar/GenDatasets/vllm_llama3_70b_eval_shots=3')
    # iter_files(root_directory)
    
    print(analyze_functions('functions.jsonl'))

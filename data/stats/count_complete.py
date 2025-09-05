import os
import re

HOME_PATH = os.path.expanduser('~')


redundant_line_comments_c = re.compile("\/\/.*")
redundant_multiline_comments_c = re.compile("\/\*.*?\*\/", re.MULTILINE|re.DOTALL)

def remove_comments(code):
    code = redundant_line_comments_c.sub("\n", code)
    code = redundant_multiline_comments_c.sub("\n", code)

    return code


def find_files_and_word_count(root_dir, file_name="pred.cpp"):
    count, total = 0, 0

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname == file_name:
                file_path = os.path.join(dirpath, fname)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        code = remove_comments(content)

                        count += '...' in code
                        total += 1
                        
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    
    return count, total


if __name__ == "__main__":
    root_directory = os.path.join(HOME_PATH, 'UniPar/GenDatasets/vllm_llama3_70b_eval_shots=3')
    count, total = find_files_and_word_count(root_directory, file_name="pred_0.cpp")
    
    print(count, total)

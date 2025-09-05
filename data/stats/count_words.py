import os
import numpy as np

HOME_PATH = os.path.expanduser('~')


def find_files_and_word_count(root_dir, file_name="pred.cpp"):
    """
    Recursively find all files with the specified name and compute their word counts.
    
    Parameters:
        root_dir (str): The root directory to start the search.
        file_name (str): The name of the file to search for (default is 'pred.cpp').
    
    Returns:
        list: A list of word counts for each found file.
    """
    word_counts = []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname == file_name:
                file_path = os.path.join(dirpath, fname)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        word_count = len(content.split())
                        word_counts.append(word_count)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    
    return word_counts

def calculate_statistics(word_counts):
    """
    Calculate the mean and standard deviation of word counts.
    
    Parameters:
        word_counts (list): List of word counts for each file.
    
    Returns:
        tuple: Mean and standard deviation of the word counts.
    """
    if not word_counts:
        return None, None  # No files found
    
    mean = np.mean(word_counts)
    std = np.std(word_counts)
    return mean, std

if __name__ == "__main__":
    root_directory = os.path.join(HOME_PATH, 'UniPar/GenDatasets/gpt-4o-mini_eval_shots=0')
    word_counts = find_files_and_word_count(root_directory, file_name="truth.cpp")
    
    if word_counts:
        mean, std = calculate_statistics(word_counts)
        print(f"Mean word count: {mean:.2f}")
        print(f"Standard deviation of word count: {std:.2f}")
    else:
        print("No files named 'pred.cpp' found.")

# TRUTH
# Mean word count: 1060.29
# Standard deviation of word count: 929.36

# GPT 0
# Mean word count: 890.20
# Standard deviation of word count: 502.59

# LLAMA 8 0
# Mean word count: 1722.76
# Standard deviation of word count: 3755.29

# LLAMA 70 0
# Mean word count: 874.83
# Standard deviation of word count: 518.19

# LLAMA 70 3
# Mean word count: 809.10
# Standard deviation of word count: 469.41
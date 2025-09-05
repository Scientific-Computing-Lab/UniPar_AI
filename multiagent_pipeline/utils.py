import re
import os
import logging
import json
from datetime import datetime


def extract_code_from_model_output(text):
    # Try to extract code between ```c++ ... ```
    match = re.search(r"```(?:\w+\n)?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try to find the first #include and return from there
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("#include"):
            return "\n".join(lines[i:]).strip()
    return text.strip()


def detect_suspicious_code(code, context=None):

    return False, None


def save_suspicious_example(code, reason, context=None):
    """
    Save suspicious code example for later review.

    Args:
        code (str): The suspicious code
        reason (str): The reason why the code is suspicious
        context (dict, optional): Additional context about the code
    """
    save_dir = os.path.join(os.getcwd(), "suspicious_examples")
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a unique filename based on context or timestamp
    if context and 'kernel_name' in context:
        filename = f"{context['kernel_name']}_{timestamp}"
    else:
        filename = f"suspicious_{timestamp}"

    # Save the code
    with open(os.path.join(save_dir, f"{filename}.cpp"), "w") as f:
        f.write(code)

    # Save metadata including the reason and context
    metadata = {
        "reason": reason,
        "timestamp": timestamp,
    }
    if context:
        metadata.update(context)

    with open(os.path.join(save_dir, f"{filename}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logging.warning(f"Suspicious code detected: {reason}. Saved to {save_dir}/{filename}.cpp")

def copy_code(code, dst, filename):
    """
    Copy the translated code to the destination directory.
    """
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, filename), 'w') as f:
        f.write(code)

def extract_run_args(makefile_path):
    with open(makefile_path, 'r') as f:
        content = f.read()

    # Match the run target and the command after it
    run_match = re.search(r'run\s*:\s*\$\(program\)\s*\n\s*(.+)', content)
    if not run_match:
        return None

    run_command = run_match.group(1)

    # Extract everything after the executable call
    args_match = re.search(r'\./\$\(program\)\s+(.*)', run_command)
    if args_match:
        return args_match.group(1).strip()

    return None

def save_fixed_examples(errors, codes, kernel_name, from_api, to_api, exec_name, code, save_dir):
    save_dir = os.path.join(os.getcwd(), save_dir)
    print(f"Saving fixed examples to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    for i, (err, c) in enumerate(zip(errors, codes), 1):
        with open(os.path.join(save_dir, f"{kernel_name}_{from_api}_{to_api}_attempt{i}_{exec_name}"), "w") as f:
            f.write(c)
        with open(os.path.join(save_dir, f"{kernel_name}_{from_api}_{to_api}_attempt{i}_error.txt"), "w") as f:
            f.write(err)
        with open(os.path.join(save_dir, f"{kernel_name}_{from_api}_{to_api}_final_{exec_name}"), "w") as f:
            f.write(code)

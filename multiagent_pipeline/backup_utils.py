"""
Utility functions for creating backups of code and errors, and generating result files.
"""
import os
import logging
import re

def create_code_backup(source_file, destination_dir, stage, attempt_num):
    """
    Creates a backup of the source code file with the naming convention pred_compile_X or pred_run_X.

    Args:
        source_file (str): Path to the source code file
        destination_dir (str): Directory to save the backup
        stage (str): 'compile' or 'run'
        attempt_num (int): Attempt number

    Returns:
        str: Path to the backup file
    """
    if not os.path.exists(source_file):
        logging.error(f"Source file {source_file} does not exist")
        return None

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Determine the file extension
    _, ext = os.path.splitext(source_file)
    if not ext:
        # Default to .cpp if no extension
        ext = '.cpp'

    # Create the backup filename
    backup_filename = f"pred_{stage}_{attempt_num}{ext}"
    backup_path = os.path.join(destination_dir, backup_filename)

    # Copy the file
    try:
        with open(source_file, 'r') as src_file:
            code = src_file.read()

        with open(backup_path, 'w') as dst_file:
            dst_file.write(code)

        logging.info(f"Created backup of {source_file} at {backup_path}")
        return backup_path
    except Exception as e:
        logging.error(f"Error creating backup of {source_file}: {str(e)}")
        return None

def create_error_backup(error_message, destination_dir, stage, attempt_num):
    """
    Creates a backup of the error message with a naming convention that links it to the corresponding code backup.

    Args:
        error_message (str): Error message to save
        destination_dir (str): Directory to save the backup
        stage (str): 'compile' or 'run'
        attempt_num (int): Attempt number

    Returns:
        str: Path to the error file
    """
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Create the error filename
    error_filename = f"error_{stage}_{attempt_num}.txt"
    error_path = os.path.join(destination_dir, error_filename)

    # Save the error message
    try:
        with open(error_path, 'w') as f:
            f.write(error_message)

        logging.info(f"Created error backup at {error_path}")
        return error_path
    except Exception as e:
        logging.error(f"Error creating error backup: {str(e)}")
        return None

def check_for_pass(output):
    """
    Checks if the word 'pass' appears in the output.

    Args:
        output (str): Output to check

    Returns:
        bool: True if 'pass' appears in the output, False otherwise
    """
    return output and 'pass' in output.lower()

def generate_result_file(kernel_dir, compile_attempt, compile_error, run_attempt, run_output, pass_status):
    """
    Creates a result.txt file with information about compilation and execution.

    Args:
        kernel_dir (str): Directory for the kernel
        compile_attempt (int): Which model-assisted attempt compiled successfully (0 for initial code)
        compile_error (str): Compilation error if compilation failed
        run_attempt (int): Which model-assisted attempt ran successfully (0 for initial code)
        run_output (str): Output from the run
        pass_status (bool): Whether the word 'pass' appears in the run output

    Returns:
        str: Path to the result file
    """
    # Create the result filename
    result_path = os.path.join(kernel_dir, "result.txt")

    # Generate the result content
    content = []
    content.append("=== COMPILATION RESULT ===")
    if compile_attempt is not None:
        if compile_attempt == 0:
            content.append("Initial code compiled successfully.")
        else:
            content.append(f"Model-assisted attempt {compile_attempt} compiled successfully.")
    else:
        content.append("Compilation failed.")
        if compile_error:
            content.append("\nCompilation Error:")
            content.append(compile_error)

    content.append("\n=== EXECUTION RESULT ===")
    if run_attempt is not None:
        if run_attempt == 0:
            content.append("Initial code ran successfully.")
        else:
            content.append(f"Model-assisted attempt {run_attempt} ran successfully.")

        if run_output:
            content.append("\nExecution Output:")
            content.append(run_output)

        content.append("\n=== VALIDATION RESULT ===")
        if pass_status:
            content.append("PASS: Found 'pass' in the run output.")
        else:
            content.append("FAIL: Did not find 'pass' in the run output.")
    else:
        content.append("Execution failed or was not attempted.")

    # Save the result file
    try:
        with open(result_path, 'w') as f:
            f.write('\n'.join(content))

        logging.info(f"Generated result file at {result_path}")
        return result_path
    except Exception as e:
        logging.error(f"Error generating result file: {str(e)}")
        return None

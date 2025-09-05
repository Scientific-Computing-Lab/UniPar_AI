#!/usr/bin/env python3
"""
Script to parse result.txt files from a folder in to_analyse and generate a summary file and CSV.
"""
import os
import sys
import argparse
import logging
import re
import csv
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_for_pass(output):
    """
    Checks if the word 'pass' appears in the output.

    Args:
        output (str): Output to check

    Returns:
        bool: True if 'pass' appears in the output, False otherwise
    """
    return output and 'pass' in output.lower()

def parse_result_file(result_file):
    """
    Parses a result.txt file to extract compilation and execution information.

    Args:
        result_file (str): Path to the result.txt file

    Returns:
        tuple: (compile_status, compile_attempt, run_status, pass_status)
    """
    try:
        with open(result_file, 'r') as f:
            result_content = f.read()

        # Check if compilation was successful
        compile_status = "Failed"
        compile_attempt = "N/A"

        if "Initial code compiled successfully" in result_content:
            compile_status = "Success"
            compile_attempt = "Initial code (0)"
        elif "Model-assisted attempt" in result_content:
            match = re.search(r"Model-assisted attempt (\d+) compiled successfully", result_content)
            if match:
                compile_status = "Success"
                compile_attempt = f"Attempt {match.group(1)}"

        # Check if execution was successful
        run_status = "Failed"
        pass_status = "N/A"

        if "Initial code ran successfully" in result_content:
            run_status = "Success"
            # Check if the word 'pass' appears in the run output
            if "PASS: Found 'pass' in the run output" in result_content:
                pass_status = "Pass"
            else:
                pass_status = "Fail"
        elif "Model-assisted attempt" in result_content:
            match = re.search(r"Model-assisted attempt (\d+) ran successfully", result_content)
            if match:
                run_status = "Success"
                # Check if the word 'pass' appears in the run output
                if "PASS: Found 'pass' in the run output" in result_content:
                    pass_status = "Pass"
                else:
                    pass_status = "Fail"

        return (compile_status, compile_attempt, run_status, pass_status)
    except Exception as e:
        logging.error(f"Error parsing result file {result_file}: {str(e)}")
        return ("Error", "Error", "Error", "Error")

def generate_summary_file(input_folder, output_file):
    """
    Generates a summary file with results for each kernel.

    Args:
        input_folder (str): Path to the input folder
        output_file (str): Path to the output summary file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get the list of API combinations in the input folder
        api_combinations = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]

        # Filter out folders that don't have a target API of OMP or CUDA
        target_apis = ["omp", "cuda"]
        api_combinations = [api for api in api_combinations if any(api.endswith(f"-{target}") for target in target_apis)]

        if not api_combinations:
            logging.error(f"No API combinations with target API of OMP or CUDA found in {input_folder}")
            return False

        # Generate the summary content
        content = []
        content.append("=== SUMMARY OF KERNEL RESULTS ===")
        content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"Input folder: {input_folder}")
        content.append("")

        # Track overall statistics
        total_kernels = 0
        compiled_kernels = 0
        ran_kernels = 0
        passed_kernels = 0

        # Process each API combination
        for api_combination in sorted(api_combinations):
            # Extract source and target APIs from the combination name
            match = re.match(r"HeCBench-(\w+)-(\w+)", api_combination)
            if not match:
                logging.warning(f"Skipping directory with invalid format: {api_combination}")
                continue

            source_api = match.group(1)
            target_api = match.group(2)

            # Skip if target API is not OMP or CUDA
            if target_api.lower() not in target_apis:
                logging.info(f"Skipping {api_combination} as target API is not OMP or CUDA")
                continue

            content.append(f"\nAPI Combination: {source_api} -> {target_api}")

            # Get the list of kernels for this API combination
            api_dir = os.path.join(input_folder, api_combination)
            src_dir = os.path.join(api_dir, "src")
            if not os.path.exists(src_dir):
                content.append(f"  No src directory found for {api_combination}")
                continue

            # Get the list of kernels
            kernels = [k for k in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, k))]
            if not kernels:
                content.append(f"  No kernels found for {api_combination}")
                continue

            # Track API-specific statistics
            api_total_kernels = len(kernels)
            api_compiled_kernels = 0
            api_ran_kernels = 0
            api_passed_kernels = 0

            content.append(f"  Found {api_total_kernels} kernels")
            total_kernels += api_total_kernels

            # Process each kernel
            for kernel in sorted(kernels):
                kernel_dir = os.path.join(src_dir, kernel)

                # Check if the result.txt file exists
                result_file = os.path.join(kernel_dir, 'result.txt')
                if not os.path.exists(result_file):
                    content.append(f"  {kernel}: No result file found")
                    continue

                # Parse the result.txt file
                compile_status, compile_attempt, run_status, pass_status = parse_result_file(result_file)

                # Update statistics
                if compile_status == "Success":
                    api_compiled_kernels += 1
                    compiled_kernels += 1

                if run_status == "Success":
                    api_ran_kernels += 1
                    ran_kernels += 1

                if pass_status == "Pass":
                    api_passed_kernels += 1
                    passed_kernels += 1

                # Add kernel result to summary
                content.append(f"  {kernel}: Compilation: {compile_status} ({compile_attempt}), Execution: {run_status}, Validation: {pass_status}")

            # Add API-specific statistics
            content.append(f"  API Statistics: {api_compiled_kernels}/{api_total_kernels} compiled, {api_ran_kernels}/{api_total_kernels} ran, {api_passed_kernels}/{api_total_kernels} passed")

        # Add overall statistics
        content.append("\n=== OVERALL STATISTICS ===")
        content.append(f"Total kernels: {total_kernels}")
        content.append(f"Compiled kernels: {compiled_kernels}/{total_kernels} ({compiled_kernels/total_kernels*100:.2f}%)")
        content.append(f"Ran kernels: {ran_kernels}/{total_kernels} ({ran_kernels/total_kernels*100:.2f}%)")
        content.append(f"Passed kernels: {passed_kernels}/{total_kernels} ({passed_kernels/total_kernels*100:.2f}%)")

        # Write the summary file
        with open(output_file, 'w') as f:
            f.write('\n'.join(content))

        logging.info(f"Generated summary file at {output_file}")
        return True
    except Exception as e:
        logging.error(f"Error generating summary file: {str(e)}")
        return False

def generate_csv_file(input_folder, output_file):
    """
    Generates a CSV file with results for each kernel.

    Args:
        input_folder (str): Path to the input folder
        output_file (str): Path to the output CSV file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get the list of API combinations in the input folder
        api_combinations = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]

        # Filter out folders that don't have a target API of OMP or CUDA
        target_apis = ["omp", "cuda"]
        api_combinations = [api for api in api_combinations if any(api.endswith(f"-{target}") for target in target_apis)]

        if not api_combinations:
            logging.error(f"No API combinations with target API of OMP or CUDA found in {input_folder}")
            return False

        # Define the CSV headers
        headers = ["Kernel", "Source API", "Target API", "Compilation Status", "Compilation Attempt", "Execution Status", "Validation Status"]

        # Collect data for the CSV
        csv_data = []

        # Process each API combination
        for api_combination in sorted(api_combinations):
            # Extract source and target APIs from the combination name
            match = re.match(r"HeCBench-(\w+)-(\w+)", api_combination)
            if not match:
                logging.warning(f"Skipping directory with invalid format: {api_combination}")
                continue

            source_api = match.group(1)
            target_api = match.group(2)

            # Skip if target API is not OMP or CUDA
            if target_api.lower() not in target_apis:
                logging.info(f"Skipping {api_combination} as target API is not OMP or CUDA")
                continue

            # Get the list of kernels for this API combination
            api_dir = os.path.join(input_folder, api_combination)
            src_dir = os.path.join(api_dir, "src")
            if not os.path.exists(src_dir):
                logging.warning(f"No src directory found for {api_combination}")
                continue

            # Get the list of kernels
            kernels = [k for k in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, k))]
            if not kernels:
                logging.warning(f"No kernels found for {api_combination}")
                continue

            # Process each kernel
            for kernel in sorted(kernels):
                kernel_dir = os.path.join(src_dir, kernel)

                # Check if the result.txt file exists
                result_file = os.path.join(kernel_dir, 'result.txt')
                if not os.path.exists(result_file):
                    # If no result file, add a row with "Not Processed" status
                    csv_data.append([kernel, source_api, target_api, "Not Processed", "N/A", "Not Processed", "N/A"])
                    continue

                # Parse the result.txt file
                compile_status, compile_attempt, run_status, pass_status = parse_result_file(result_file)

                # Add a row to the CSV data
                csv_data.append([kernel, source_api, target_api, compile_status, compile_attempt, run_status, pass_status])

        # Write the CSV file
        with open(output_file, 'w', newline='') as f:
            # Write the header line manually to ensure exact format
            f.write(','.join(headers) + '\n')
            # Use csv writer for the data rows
            writer = csv.writer(f)
            writer.writerows(csv_data)

        logging.info(f"Generated CSV file at {output_file}")
        return True
    except Exception as e:
        logging.error(f"Error generating CSV file: {str(e)}")
        return False

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Parse result.txt files and generate summary and CSV files.')
    parser.add_argument('--input_folder', dest='input_folder', help='Path to the input folder (e.g., multiagent_pipeline/to_analyse/eval_shot_0_m_15000_t_0.2_p_0.9_new)', default='./to_analyse/eval_shot_0_m_15000_t_0.2_p_0.9_new')
    parser.add_argument('--summary', help='Path to the output summary file (default: summary.txt in the input folder)', default=None)
    parser.add_argument('--csv', help='Path to the output CSV file (default: kernel_results.csv in the input folder)', default=None)

    return parser.parse_args()

def main():
    """
    Main function to parse result.txt files and generate summary and CSV files.
    """
    args = parse_arguments()

    # Validate input folder
    if not os.path.exists(args.input_folder):
        logging.error(f"Input folder {args.input_folder} does not exist")
        return 1

    # Set default output files if not specified
    summary_file = args.summary or os.path.join(args.input_folder, "summary.txt")
    csv_file = args.csv or os.path.join(args.input_folder, "kernel_results.csv")

    # Generate summary file
    if not generate_summary_file(args.input_folder, summary_file):
        logging.error("Failed to generate summary file")
        return 1

    # Generate CSV file
    if not generate_csv_file(args.input_folder, csv_file):
        logging.error("Failed to generate CSV file")
        return 1

    logging.info("Successfully generated summary and CSV files")
    return 0

if __name__ == "__main__":
    sys.exit(main())

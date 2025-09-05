#!/usr/bin/env python3
"""
Main script for running the UniPar code rewrite using the datasets_after_eval_compile folder.
This script uses the eval compile methods and the dataset in the datasets_after_eval_compile folder
as the existing dataset for the solver.
"""
import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

from model_agent import ModelAgent
from solver import ExecutionAgent
from executioner import RunnerAgent
from backup_utils import check_for_pass, generate_result_file

# Set up logging - will be reconfigured after output directory is created
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run UniPar code rewrite using datasets_after_eval_compile')

    parser.add_argument('--dataset_dir', type=str, default='datasets_after_eval_compile/shot_0_m_15000_t_0.2_p_0.9',
                        help='Path to the datasets_after_eval_compile directory (default: datasets_after_eval_compile/shot_0_m_15000_t_0.2_p_0.9)')
    parser.add_argument('--from_api', type=str, required=True,
                        help='Source API (e.g., cuda, omp, serial, hip, sycl, all)')
    parser.add_argument('--to_api', type=str, required=True,
                        help='Target API (e.g., cuda, omp, serial, hip, sycl)')
    parser.add_argument('--hecbench_path', type=str, default=None,
                        help='Path to the HeCBench directory (default: multiagent_pipeline/hecbench_omp)')
    parser.add_argument('--max_compile_attempts', type=int, default=3,
                        help='Maximum number of compilation attempts (default: 3)')
    parser.add_argument('--max_run_attempts', type=int, default=3,
                        help='Maximum number of run attempts (default: 3)')
    parser.add_argument('--kernel', type=str, default=None,
                        help='Specific kernel to process (default: process all kernels)')
    parser.add_argument('--model_name', type=str, default='gpt-4',
                        help='Model to use for code generation (default: gpt-4)')

    return parser.parse_args()

def main():
    """Main function to run the UniPar code rewrite using datasets_after_eval_compile."""
    args = parse_arguments()

    # Set up paths
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create a timestamp for the output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_path, 'output', f"{timestamp}_datasets_after_eval_compile")
    os.makedirs(output_dir, exist_ok=True)

    # Reconfigure logging to include a file in the output directory
    log_file = os.path.join(output_dir, "datasets_after_eval_compile.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.info(f"Log file created at {log_file}")

    # First, run eval_compile to create a clean dataset
    logging.info("Running eval_compile to create a clean dataset...")

    # Set up paths for eval_compile
    existing_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dataset_dir)
    if not os.path.exists(existing_dataset_path):
        logging.error(f"Existing dataset directory {existing_dataset_path} does not exist")
        return 1

    # Set up HeCBench path
    if args.hecbench_path:
        hecbench_path = args.hecbench_path
    else:
        hecbench_path = '/home/tomerbitan/unipar/HeCBench'#os.path.join(project_path, 'multiagent_pipeline', 'hecbench_omp')

    if not os.path.exists(hecbench_path):
        logging.error(f"HeCBench directory {hecbench_path} does not exist")
        return 1

    # Extract model name from the original dataset directory
    dataset_dir_parts = args.dataset_dir.split('/')
    original_model_name = "unknown"

    # Check if the dataset directory contains a model name
    if len(dataset_dir_parts) > 0:
        last_part = dataset_dir_parts[-1]
        # Try to extract model name from directory name
        if "vllm_" in last_part:
            # Format like "vllm_llama3_70b_eval_shots=0_max_token=5000.0_temp=0.6_p=0.8"
            model_parts = last_part.split('_shots=')
            if len(model_parts) > 0:
                original_model_name = model_parts[0]
        elif "shot_" in last_part:
            # Format like "shot_0_m_15000_t_0.2_p_0.9"
            original_model_name = "gpt"  # Default to gpt if no specific model name found

    logging.info(f"Original model name extracted: {original_model_name}")

    # Create a clean dataset directory within the output directory
    # Include both the original model name and the solver model name in the directory name
    clean_dataset_dir = args.dataset_dir #os.path.join(output_dir, f'clean_dataset_{original_model_name}_solver_{args.model_name}')
    # os.makedirs(clean_dataset_dir, exist_ok=True)

    # Determine which source APIs to process
    source_apis = []
    if args.from_api.lower() == "all":
        # Use all available source APIs
        source_apis = ['serial', 'omp', 'cuda']#, 'hip', 'sycl']
    else:
        source_apis = [args.from_api]

    #
    # Use the clean dataset as the input dataset for the main processing
    dataset_path = clean_dataset_dir
    logging.info(f"Using clean dataset at {dataset_path} for main processing")

    # Initialize the model agent
    # Use a local model server to avoid authentication issues
    model_agent = ModelAgent(
        model=args.model_name,
        api_base='http://localhost:8000/v1',
        api_key=""#'mult_pipeline'
    )

    # Initialize the execution agent and runner agent
    execution_agent = ExecutionAgent(model_agent)
    runner_agent = RunnerAgent(model_agent)

    # Determine which source APIs to process
    source_apis = []
    if args.from_api.lower() == "all":
        # Find all available source APIs in the dataset directory
        for item in os.listdir(dataset_path):
            if item.startswith("HeCBench-") and item.endswith(f"-{args.to_api}"):
                api = item.split("-")[1]
                source_apis.append(api)

        if not source_apis:
            logging.error(f"No source APIs found for target API {args.to_api}")
            return 1

        logging.info(f"Found {len(source_apis)} source APIs to process: {', '.join(source_apis)}")
    else:
        source_apis = [args.from_api]

    # Process each source API
    for from_api in source_apis:
        logging.info(f"Processing source API: {from_api} to {args.to_api}")

        # Get the list of kernels to process
        hecbench_dir = os.path.join(dataset_path, f"HeCBench-{from_api}-{args.to_api}")
        if not os.path.exists(hecbench_dir):
            logging.error(f"HeCBench directory {hecbench_dir} does not exist")
            continue

        src_dir = os.path.join(hecbench_dir, "src")
        if not os.path.exists(src_dir):
            logging.error(f"src directory {src_dir} does not exist")
            continue

        if args.kernel:
            # Process a specific kernel
            kernels = [k for k in os.listdir(src_dir) if k.startswith(args.kernel)]
        else:
            # Process all kernels
            kernels = os.listdir(src_dir)

        if not kernels:
            logging.error(f"No kernels found for {from_api} to {args.to_api}")
            continue

        logging.info(f"Found {len(kernels)} kernels to process for {from_api} to {args.to_api}")

        # Process each kernel
        for kernel in kernels:
            logging.info(f"Processing kernel: {kernel}")

            kernel_dir = os.path.join(src_dir, kernel)
            if not os.path.isdir(kernel_dir):
                logging.warning(f"Skipping {kernel}: not a directory")
                continue
            else:
                files = os.listdir(kernel_dir)
                if "pred_0.cpp" in files or "pred_run_0.cpp" in files:
                    logging.info(f"Skipping kernel {kernel} as it already has prediction files.")
                    continue
            
            # Create a compilation directory for this kernel
            compile_dir = os.path.join(src_dir, kernel)
            # os.makedirs(compile_dir, exist_ok=True)

            # In the new structure, we need to find the main source file in the kernel directory
            # The main file is typically named after the kernel with a .cu or .cpp extension
            main_files = [f for f in os.listdir(kernel_dir) if f.endswith('.cu') or f.endswith('.cpp')]
            if not main_files:
                logging.warning(f"Skipping {kernel}: no source files found")
                continue

            # Use the first source file as the initial code file
            initial_code_file = os.path.join(kernel_dir, main_files[0])

            # Extract kernel name and APIs from the directory name
            # The kernel directory name is in the format "kernel-api"
            kernel_parts = kernel.split('-')
            kernel_name = kernel_parts[0]
            to_api = args.to_api

            # Compile the code
            logging.info(f"Compiling {kernel}")
            compile_result = execution_agent.compile_and_run(
                build_dir=compile_dir,
                exec_name=os.path.basename(initial_code_file),
                kernel_name=kernel_name,
                max_attempts=args.max_compile_attempts,
                from_api=from_api,
                to_api=to_api,
                initial_dir=kernel_dir
            )

            if not compile_result['success']:
                logging.error(f"Compilation failed for {kernel}: {compile_result['error']}")

                # Generate result.txt file with compilation error
                generate_result_file(
                    kernel_dir=kernel_dir,
                    compile_attempt=None,  # Compilation failed
                    compile_error=compile_result['error'],
                    run_attempt=None,  # No run attempt
                    run_output=None,  # No run output
                    pass_status=False  # No pass status
                )
                continue

            # Run the code
            logging.info(f"Running {kernel}")
            run_result = runner_agent.run_program(
                build_dir=compile_dir,
                exec_name=os.path.basename(initial_code_file),
                kernel_name=kernel_name,
                max_attempts=args.max_run_attempts,
                from_api=from_api,
                to_api=to_api,
                kernel_dir=kernel_dir
            )

            # Generate result.txt file with compilation and run information
            compile_attempt = 0 if compile_result['compile_iteration'] == 0 else compile_result['compile_iteration']

            if not run_result['success']:
                logging.error(f"Execution failed for {kernel}: {run_result['error']}")
                generate_result_file(
                    kernel_dir=kernel_dir,
                    compile_attempt=compile_attempt,
                    compile_error=None,  # Compilation was successful
                    run_attempt=None,  # Run failed
                    run_output=run_result['error'],  # Use error as run output
                    pass_status=False  # No pass status
                )
            else:
                logging.info(f"Execution succeeded for {kernel}")
                run_attempt = 0 if run_result['execution_iteration'] == 0 else run_result['execution_iteration']

                # Check if the results are valid
                pass_status = check_for_pass(run_result['output'])

                if run_result['results_valid']:
                    logging.info(f"Results are valid for {kernel}: {run_result['validation_message']}")
                else:
                    logging.warning(f"Results are not valid for {kernel}: {run_result['validation_message']}")

                generate_result_file(
                    kernel_dir=kernel_dir,
                    compile_attempt=compile_attempt,
                    compile_error=None,  # Compilation was successful
                    run_attempt=run_attempt,
                    run_output=run_result['output'],
                    pass_status=pass_status
                )

    logging.info(f"Processing complete. Output directory: {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

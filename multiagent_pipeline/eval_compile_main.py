"""
Main script for the UniPar code rewrite using the eval compile methods and the datasets_after_eval_compile folder.
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
from eval_compile import update_hecbench
from backup_utils import check_for_pass, generate_result_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='UniPar code rewrite using eval compile methods')

    parser.add_argument('--from_api', type=str, default='omp', required=False, choices=['serial', 'omp', 'cuda', 'hip', 'sycl'],
                        help='Source API (e.g., serial, omp, cuda)')
    parser.add_argument('--to_api', type=str, default='cuda', required=False, choices=['omp', 'cuda'],
                        help='Target API (e.g., omp, cuda)')
    parser.add_argument('--dataset_dir', type=str, default='datasets_after_eval_compile/shot_0_m_15000_t_0.2_p_0.9',
                        help='Directory containing the dataset (default: datasets_after_eval_compile)')
    parser.add_argument('--hecbench_path', type=str, default=None,
                        help='Path to the HeCBench directory (default: determined from project path)')
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
    """Main function to run the UniPar code rewrite."""
    args = parse_arguments()

    # Set up paths
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dataset_dir)

    if not os.path.exists(dataset_path):
        logging.error(f"Dataset directory {dataset_path} does not exist")
        return 1

    # Set up HeCBench path
    if args.hecbench_path:
        hecbench_path = args.hecbench_path
    else:
        hecbench_path = os.path.join(project_path, 'multiagent_pipeline', 'hecbench_omp')

    if not os.path.exists(hecbench_path):
        logging.error(f"HeCBench directory {hecbench_path} does not exist")
        return 1

    # Create a timestamp for the output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_path, 'output', f"{timestamp}_eval_compile")
    os.makedirs(output_dir, exist_ok=True)

    # Create a temporary directory for HeCBench
    temp_dir = os.path.join(output_dir, f"HeCBench-{args.from_api}-{args.to_api}")
    os.makedirs(temp_dir, exist_ok=True)

    # Initialize the model agent
    # Use a local model server to avoid authentication issues
    model_agent = ModelAgent(
        model=args.model_name,
        api_base='http://localhost:8000/v1',
        api_key='mult_pipeline'
    )

    # Initialize the execution agent and runner agent
    execution_agent = ExecutionAgent(model_agent)
    runner_agent = RunnerAgent(model_agent)

    # Get the list of kernels to process
    # The dataset structure is different than expected
    # The kernels are in the 'src' directory of the HeCBench directory
    hecbench_dir = os.path.join(dataset_path, f"HeCBench-{args.from_api}-{args.to_api}")
    if not os.path.exists(hecbench_dir):
        logging.error(f"HeCBench directory {hecbench_dir} does not exist")
        return 1

    src_dir = os.path.join(hecbench_dir, "src")
    if not os.path.exists(src_dir):
        logging.error(f"src directory {src_dir} does not exist")
        return 1

    if args.kernel:
        # Process a specific kernel
        kernels = [k for k in os.listdir(src_dir) if k.startswith(args.kernel)]
    else:
        # Process all kernels
        kernels = os.listdir(src_dir)

    if not kernels:
        logging.error(f"No kernels found for {args.from_api} to {args.to_api}")
        return 1

    logging.info(f"Found {len(kernels)} kernels to process")

    # Process each kernel
    for kernel in kernels:
        logging.info(f"Processing kernel: {kernel}")

        kernel_dir = os.path.join(src_dir, kernel)
        if not os.path.isdir(kernel_dir):
            logging.warning(f"Skipping {kernel}: not a directory")
            continue

        # Create a compilation directory for this kernel
        compile_dir = os.path.join(output_dir, 'compilation', f"{kernel}")
        os.makedirs(compile_dir, exist_ok=True)

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
        from_api = args.from_api
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

        if not run_result['success']:
            logging.error(f"Execution failed for {kernel}: {run_result['error']}")
        else:
            logging.info(f"Execution succeeded for {kernel}")

            # Check if the results are valid
            if run_result['results_valid']:
                logging.info(f"Results are valid for {kernel}: {run_result['validation_message']}")
            else:
                logging.warning(f"Results are not valid for {kernel}: {run_result['validation_message']}")

    # Skip updating the HeCBench directory for now
    logging.info(f"Skipping update of HeCBench directory")
    # The update_hecbench function expects a different directory structure
    # and would require significant changes to work with the new structure
    # update_hecbench(
    #     from_api=args.from_api,
    #     to_api=args.to_api,
    #     gen_code_path=output_dir,
    #     hecbench_path=hecbench_path,
    #     temp_dir=temp_dir
    # )

    logging.info(f"Processing complete. Output directory: {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

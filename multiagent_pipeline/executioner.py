import subprocess
from utils import *
from solver import ExecutionAgent as SolverExecutionAgent
from backup_utils import check_for_pass, create_code_backup, create_error_backup, generate_result_file
import sys
sys.path.append('/home/tomerbitan/unipar/UniPar/')
from replace_main import replace_with_original
hec_bench_loc = '/home/tomerbitan/unipar/HeCBench/src/'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RunnerAgent:
    def __init__(self, model_agent):
        self.model_agent = model_agent
        self.solver_agent = SolverExecutionAgent(model_agent)
        self.program_results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'program_results')

    def verify_program_results(self, kernel_name, to_api, output, build_dir=None):
        """
        Verify that the program results contain "PASS" in the result.txt file.

        Args:
            kernel_name (str): The name of the kernel/program
            to_api (str): The target API (e.g., 'omp', 'cuda')
            output (str): The output of the program execution
            build_dir (str, optional): The directory where the program was built and run

        Returns:
            tuple: (is_valid, message) where is_valid is a boolean indicating if the results are valid
                  and message is a string with details about the validation
        """
        try:
            # If build_dir is provided, look for result.txt in that directory
            if build_dir and os.path.isdir(build_dir):
                results_file = os.path.join(build_dir, "result.txt")

                # Check if result.txt exists
                if os.path.exists(results_file):
                    # Read the file and check for "pass" (case-insensitive)
                    with open(results_file, 'r') as f:
                        content = f.read()
                        if "pass" in content.lower():
                            return True, "Found 'pass' in result.txt"
                        else:
                            return False, "Did not find 'pass' in result.txt"
                else:
                    return False, f"Results file not found: {results_file}"
            else:
                # If build_dir is not provided or not a directory, check the output for "pass" (case-insensitive)
                if "pass" in output.lower():
                    return True, "Found 'pass' in program output"
                else:
                    return False, "Did not find 'pass' in program output"

        except Exception as e:
            return False, f"Error verifying program results: {str(e)}"

    def run_program(self, build_dir, exec_name, kernel_name, max_attempts, from_api, to_api, kernel_dir=None):
        attempt = 0
        codes = []
        errors = []
        code = ""  # Initialize code with a default empty string
        runtime_attempts_base_dir = build_dir
        if to_api == 'omp':
            makefile_path = os.path.join(build_dir, "Makefile.aomp")
        elif to_api == 'cuda':
            makefile_path = os.path.join(build_dir, "Makefile")
        try:
            input = extract_run_args(makefile_path)
        except Exception as e:
            logging.error(f"Error extracting run arguments: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_iteration': attempt + 1,
                'translated_code': None,
                'results_valid': False,
                'validation_message': f'Error extracting run arguments: {str(e)}, could not verify results'
            }
        while attempt < max_attempts:
            # Determine the current filename
            if attempt == 0:
                current_filename = exec_name  # This is pred_0.cpp or pred_0.cu for the first iteration
            else:
                # For subsequent attempts, use the main filename
                current_filename = SolverExecutionAgent.extract_source_filename_from_makefile(build_dir, to_api)

            # For the final run, determine the source filename from the Makefile
            run_filename = SolverExecutionAgent.extract_source_filename_from_makefile(build_dir, to_api)

            context = {
                "kernel_name": kernel_name,
                "from_api": from_api,
                "to_api": to_api,
                "attempt": attempt + 1,
                "exec_name": current_filename
            }
            try:
                # Compile using make
                logging.info(f"Attempting to run {current_filename} (Attempt {attempt + 1})")
                main_cpp = os.path.join(build_dir, current_filename)
                if os.path.exists(main_cpp):
                    loc_in_hec_bench ='/'.join(main_cpp.split('/')[-2:])
                    replace_with_original(hec_bench_loc+loc_in_hec_bench, main_cpp, fix_attempt=False)
                    with open(main_cpp, "r") as f:
                        code = f.read()

                    # For the first attempt, save a copy as pred_0.cu/cpp if it doesn't exist
                    # if attempt == 0 and not os.path.exists(os.path.join(build_dir, f"pred_0.{main_cpp.split('.')[-1]}")):
                    #     pred_0_path = os.path.join(build_dir, f"pred_0.{main_cpp.split('.')[-1]}")
                    #     with open(pred_0_path, "w") as f:
                    #         f.write(code)

                    # Create a backup of the code with the pred_run_X naming convention in the kernel directory
                    if kernel_dir and os.path.exists(kernel_dir):
                        create_code_backup(main_cpp, kernel_dir, 'run', attempt)
                    
                # env = os.environ.copy()  # Copy the current env
                # env["LD_LIBRARY_PATH"] = f"{os.environ['CONDA_PREFIX']}/lib:" + env.get("LD_LIBRARY_PATH", "")
                # env.update({"OMP_TARGET_OFFLOAD": "MANDATORY"})
                timeout_time = 200
                if to_api == 'omp':
                    #check if code is going to run on cpu
                    if '#pragma omp target' not in code:
                        logging.warning(f"Code does not contain OpenMP target directive, running on CPU")
                        error_message = "code does not contain OpenMP target directive, #pragma omp target, the code needs to be ran on the GPU"
                        fixed_code = self.model_agent.run_code(code, error_message, input, to_api, context=context)
                        code = extract_code_from_model_output(fixed_code)
                        if fixed_code is None:
                            logging.warning(f"Skipping {kernel_name}: {from_api}->{to_api} due to suspicious code detection")
                            # Create run-specific directory for suspicious code detection
                            run_attempt_dir = os.path.join(runtime_attempts_base_dir, f"run_{attempt}")
                            os.makedirs(run_attempt_dir, exist_ok=True)

                            # Save the suspicious code to the run attempt directory
                            with open(os.path.join(run_attempt_dir, current_filename), "w") as f:
                                f.write(code)

                            # Save the suspicious code message to the run attempt directory
                            with open(os.path.join(run_attempt_dir, "suspicious_code.txt"), "w") as f:
                                f.write("Suspicious code detected during OpenMP target directive check")

                            return {
                                'success': False,
                                'error': 'Suspicious code detected during fixing',
                                'execution_iteration': attempt + 1,
                                'translated_code': code,
                                'suspicious_code_detected': True,
                                'results_valid': False,
                                'validation_message': 'Suspicious code detected during fixing, could not verify results',
                                'attempt_dir': run_attempt_dir,
                                'runtime_attempts_base_dir': runtime_attempts_base_dir
                            }
                        with open(main_cpp, "w") as f:
                            f.write(code)
                        loc_in_hec_bench = main_cpp.split('/')[-2:]
                        replace_with_original(hec_bench_loc+loc_in_hec_bench, main_cpp, fix_attempt=False)
                        attempt += 1
                        logging.info(f"Attempting to run {exec_name} (Attempt {attempt + 1})")
                        if '#pragma omp target' not in code:
                            continue

                    compile_result = subprocess.run(
                        ['make', '-f', 'Makefile.aomp', 'run'],
                        cwd=build_dir,
                        capture_output=True,
                        text=True,
                        # env=env,
                        timeout = timeout_time
                    )

                elif to_api == 'cuda':
                    compile_result = subprocess.run(
                        ['make', 'run'],
                        cwd=build_dir,
                        capture_output=True,
                        text=True,
                        # env=env,
                        timeout = timeout_time
                    )

                if compile_result.returncode != 0:
                    if attempt == max_attempts - 1:
                        is_suspicious, reason = detect_suspicious_code(code, context)
                        if is_suspicious:
                            # Flag and save the suspicious example
                            logging.warning(f"Suspicious runtime-fixed code detected: {reason}")
                            context_with_error = context.copy() if isinstance(context, dict) else {"original_context": context}
                            context_with_error["runtime_error"] = error_message
                            context_with_error["input"] = input
                            context_with_error["suspicious_code_detected"] = True
                            code = code + f"\n\n// SUSPICIOUS CODE DETECTED: {reason}\n// This example was flagged but processing continued"
                            save_suspicious_example(code, reason, context_with_error)
                        logging.error(f"Max attempts reached. Compilation failed.")
                        break
                    error_message = compile_result.stderr
                    errors.append(error_message)
                    codes.append(code)

                    # Save the error message directly in the build directory
                    with open(os.path.join(build_dir, "run_error.txt"), "w") as f:
                        f.write(error_message)

                    # Create a backup of the error message in the kernel directory
                    if kernel_dir and os.path.exists(kernel_dir):
                        create_error_backup(error_message, kernel_dir, 'run', attempt)

                    logging.error(f"Failed to run, trying to fix the code")

                    # Pass context to the model agent for suspicious code detection
                    fixed_code = self.model_agent.run_code(code, error_message, input, to_api, context=context)
                    code = extract_code_from_model_output(fixed_code)
                    # If fixed_code is None, it means suspicious code was detected and the example should be skipped
                    if fixed_code is None:
                        logging.warning(f"Skipping {kernel_name}: {from_api}->{to_api} due to suspicious code detection")
                        return {
                            'success': False,
                            'error': 'Suspicious code detected during fixing',
                            'output': None,
                            'execution_iteration': attempt + 1,
                            'translated_code': code,
                            'suspicious_code_detected': True,
                            'results_valid': False,
                            'validation_message': 'Suspicious code detected during fixing, could not verify results'
                        }

                    if 'Error code: 400' in fixed_code:
                        break

                    # Update the main file with the fixed code
                    with open(os.path.join(build_dir, run_filename), "w") as f:
                        f.write(code)
                    loc_in_hec_bench = os.path.join(build_dir, run_filename).split('/')[-2:]
                    replace_with_original(hec_bench_loc+loc_in_hec_bench, main_cpp, fix_attempt=False)

                    attempt += 1
                    continue

                # If run succeeded, ensure we have a main.cu/main.cpp
                if current_filename != run_filename:
                    with open(os.path.join(build_dir, run_filename), "w") as f:
                        f.write(code)

                # Verify program results against expected results
                is_valid, validation_message = self.verify_program_results(kernel_name, to_api, compile_result.stdout, build_dir)
                if is_valid:
                    logging.info(f"Program results match expected results: {validation_message}")
                else:
                    logging.warning(f"Program results do not match expected results: {validation_message}")

                
                # Save the output to a file in the kernel directory
                if kernel_dir and os.path.exists(kernel_dir):
                    try:
                        # Save the output to a file
                        output_file = os.path.join(kernel_dir, f"output_run_{attempt}.txt")
                        with open(output_file, 'w') as f:
                            f.write(compile_result.stdout)

                        # Read the existing result.txt file to get compilation information
                        result_file = os.path.join(kernel_dir, "result.txt")
                        compile_attempt = None
                        if os.path.exists(result_file):
                            with open(result_file, 'r') as f:
                                content = f.read()
                                if "Initial code compiled successfully" in content:
                                    compile_attempt = 0
                                elif "Model-assisted attempt" in content:
                                    match = re.search(r"Model-assisted attempt (\d+) compiled successfully", content)
                                    if match:
                                        compile_attempt = int(match.group(1))

                        # Generate or update the result.txt file
                        generate_result_file(
                            kernel_dir,
                            compile_attempt,
                            None,  # No compile error since compilation was successful
                            attempt,
                            compile_result.stdout,
                            is_valid
                        )
                        logging.info(f"Generated/updated result.txt file in {kernel_dir}")
                    except Exception as e:
                        logging.warning(f"Error generating/updating result.txt file: {str(e)}")

                return {
                    'success': True,
                    'output': compile_result.stdout,
                    'error': compile_result.stderr,
                    'execution_iteration': attempt + 1,
                    'translated_code': code,
                    'results_valid': is_valid,
                    'validation_message': validation_message,
                    'attempt_dir': run_attempt_dir,
                    'runtime_attempts_base_dir': runtime_attempts_base_dir
                }
            except subprocess.TimeoutExpired:
                if attempt == max_attempts - 1:
                    # Last attempt - give up
                    # Create run-specific directory for the final timeout attempt
                    run_attempt_dir = os.path.join(runtime_attempts_base_dir, f"run_{attempt}")
                    os.makedirs(run_attempt_dir, exist_ok=True)

                    # Save the code that timed out to the run attempt directory
                    with open(os.path.join(run_attempt_dir, current_filename), "w") as f:
                        f.write(code)

                    # Save the timeout message to the run attempt directory
                    with open(os.path.join(run_attempt_dir, "timeout.txt"), "w") as f:
                        f.write("Execution timed out on final attempt")

                    # Extract the current results folder from build_dir
                    # build_dir format: output/{timestamp}_{dataset_name}_{solver_config}/compilation/{kernel_name}-{to_api}
                    # We need to extract {timestamp}_{dataset_name}_{solver_config}
                    parts = build_dir.split(os.sep)
                    output_index = parts.index('output') if 'output' in parts else -1
                    if output_index >= 0 and output_index + 1 < len(parts):
                        current_results_folder = parts[output_index + 1]
                        # Also save the timeout message to the kernel logs directory
                        kernel_log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                    'output', current_results_folder, f"{kernel_name}_{from_api}_to_{to_api}")
                        os.makedirs(kernel_log_dir, exist_ok=True)
                    else:
                        # Fallback to the old path if we can't extract the current results folder
                        kernel_log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                    'output', 'kernel_logs', f"{kernel_name}-{to_api}")
                        os.makedirs(kernel_log_dir, exist_ok=True)

                    # Create a new folder for this attempt within the kernel folder
                    attempt_folder = os.path.join(kernel_log_dir, f"attempt_run_{attempt}")
                    os.makedirs(attempt_folder, exist_ok=True)

                    with open(os.path.join(attempt_folder, f"timeout.txt"), "w") as f:
                        f.write(f"=== TIMEOUT ERROR (Final Attempt) ===\n")
                        f.write(f"Kernel: {kernel_name}\n")
                        f.write(f"From API: {from_api}\n")
                        f.write(f"To API: {to_api}\n")
                        f.write(f"Filename: {current_filename}\n\n")
                        f.write("=== ERROR MESSAGE ===\n")
                        f.write("Execution timed out on final attempt")

                    # Check if there's any output from previous attempts that might contain "pass"
                    output = None
                    for i in range(attempt):
                        output_file = os.path.join(runtime_attempts_base_dir, f"run_{i}", "output.txt")
                        if os.path.exists(output_file):
                            with open(output_file, 'r') as f:
                                output = f.read()
                                if "pass" in output.lower():
                                    # Found "pass" in a previous output
                                    is_valid = True
                                    validation_message = f"Found 'pass' in output from attempt {i+1}"
                                    break

                    # If no output with "pass" was found, set is_valid to False
                    if output is None or "pass" not in output.lower():
                        is_valid = False
                        validation_message = 'Execution timed out on final attempt, could not verify results'

                    return {
                        'success': False,
                        'error': 'Execution timed out on final attempt',
                        'output': output,
                        'execution_iteration': attempt + 1,
                        'translated_code': code,
                        'results_valid': is_valid,
                        'validation_message': validation_message,
                        'attempt_dir': run_attempt_dir,
                        'runtime_attempts_base_dir': runtime_attempts_base_dir
                    }
                else:
                    # Timeout but we still have more attempts - treat like a failed run
                    logging.error(f"Timeout occurred while running {current_filename} (Attempt {attempt + 1}), trying to fix code.")

                    timeout_message = "Execution timed out - probably the code is running on the CPU"
                    errors.append(timeout_message)
                    codes.append(code)

                    # Create run-specific directory
                    run_attempt_dir = os.path.join(runtime_attempts_base_dir, f"run_{attempt}")
                    os.makedirs(run_attempt_dir, exist_ok=True)

                    # Save the current file to the run attempt directory
                    with open(os.path.join(run_attempt_dir, current_filename), "w") as f:
                        f.write(code)

                    # Save the timeout message to the run attempt directory
                    with open(os.path.join(run_attempt_dir, "timeout.txt"), "w") as f:
                        f.write(timeout_message)


                    parts = build_dir.split(os.sep)
                    output_index = parts.index('output') if 'output' in parts else -1
                    if output_index >= 0 and output_index + 1 < len(parts):
                        current_results_folder = parts[output_index + 1]
                        # Also save the timeout message to the kernel logs directory
                        kernel_log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                    'output', current_results_folder, f"{kernel_name}_{from_api}_to_{to_api}")
                        os.makedirs(kernel_log_dir, exist_ok=True)
                    else:
                        # Fallback to the old path if we can't extract the current results folder
                        kernel_log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                    'output', 'kernel_logs', f"{kernel_name}-{to_api}")
                        os.makedirs(kernel_log_dir, exist_ok=True)

                    # Create a new folder for this attempt within the kernel folder
                    attempt_folder = os.path.join(kernel_log_dir, f"attempt_run_{attempt}")
                    os.makedirs(attempt_folder, exist_ok=True)

                    with open(os.path.join(attempt_folder, f"timeout.txt"), "w") as f:
                        f.write(f"=== TIMEOUT ERROR (Attempt {attempt + 1}) ===\n")
                        f.write(f"Kernel: {kernel_name}\n")
                        f.write(f"From API: {from_api}\n")
                        f.write(f"To API: {to_api}\n")
                        f.write(f"Filename: {current_filename}\n\n")
                        f.write("=== ERROR MESSAGE ===\n")
                        f.write(timeout_message)

                    # Extract the current results folder from build_dir
                    parts = build_dir.split(os.sep)
                    output_index = parts.index('output') if 'output' in parts else -1
                    current_results_folder = parts[output_index + 1] if output_index >= 0 and output_index + 1 < len(parts) else 'kernel_logs'

                    context = {
                        "kernel_name": kernel_name,
                        "from_api": from_api,
                        "to_api": to_api,
                        "attempt": attempt + 1,
                        "exec_name": current_filename,
                        "current_results_folder": current_results_folder
                    }

                    fixed_code = self.model_agent.run_code(code, timeout_message, input, to_api, context=context)

                    if fixed_code is None:
                        logging.warning(f"Skipping {kernel_name}: {from_api}->{to_api} due to suspicious code detection after timeout")
                        # Create run-specific directory for suspicious code detection
                        run_attempt_dir = os.path.join(runtime_attempts_base_dir, f"run_{attempt}")
                        os.makedirs(run_attempt_dir, exist_ok=True)

                        # Save the suspicious code to the run attempt directory
                        with open(os.path.join(run_attempt_dir, current_filename), "w") as f:
                            f.write(code)

                        # Save the suspicious code message to the run attempt directory
                        with open(os.path.join(run_attempt_dir, "suspicious_code.txt"), "w") as f:
                            f.write("Suspicious code detected during fixing after timeout")

                        # Check if there's any output from previous attempts that might contain "pass"
                        output = None
                        is_valid = False
                        validation_message = 'Suspicious code detected after timeout, could not verify results'

                        for i in range(attempt):
                            output_file = os.path.join(runtime_attempts_base_dir, f"run_{i}", "output.txt")
                            if os.path.exists(output_file):
                                with open(output_file, 'r') as f:
                                    output_content = f.read()
                                    if "pass" in output_content.lower():
                                        # Found "pass" in a previous output
                                        is_valid = True
                                        validation_message = f"Found 'pass' in output from attempt {i+1}"
                                        output = output_content
                                        break

                        return {
                            'success': False,
                            'error': 'Suspicious code detected during fixing after timeout',
                            'output': output,
                            'execution_iteration': attempt + 1,
                            'translated_code': code,
                            'suspicious_code_detected': True,
                            'results_valid': is_valid,
                            'validation_message': validation_message,
                            'attempt_dir': run_attempt_dir,
                            'runtime_attempts_base_dir': runtime_attempts_base_dir
                        }

                    code = extract_code_from_model_output(fixed_code)

                    # Write to a new file for the next iteration
                    # Use .cu extension for CUDA, .cpp for others
                    next_filename = f"pred_{attempt+1}.cu" if to_api == 'cuda' else f"pred_{attempt+1}.cpp"
                    with open(os.path.join(build_dir, next_filename), "w") as f:
                        f.write(code)

                    # Also write to main.cu/main.cpp for running
                    with open(os.path.join(build_dir, run_filename), "w") as f:
                        f.write(code)

                    attempt += 1
                    continue

            except Exception as e:
                # Create run-specific directory for exception
                run_attempt_dir = os.path.join(runtime_attempts_base_dir, f"run_{attempt}")
                os.makedirs(run_attempt_dir, exist_ok=True)

                # Save the code that caused the exception to the run attempt directory
                with open(os.path.join(run_attempt_dir, current_filename), "w") as f:
                    f.write(code)

                # Save the exception message to the run attempt directory
                with open(os.path.join(run_attempt_dir, "exception.txt"), "w") as f:
                    f.write(str(e))

                # Check if there's any output from previous attempts that might contain "pass"
                output = None
                is_valid = False
                validation_message = f'Error during execution: {str(e)}, could not verify results'

                for i in range(attempt):
                    output_file = os.path.join(runtime_attempts_base_dir, f"run_{i}", "output.txt")
                    if os.path.exists(output_file):
                        with open(output_file, 'r') as f:
                            output_content = f.read()
                            if "pass" in output_content.lower():
                                # Found "pass" in a previous output
                                is_valid = True
                                validation_message = f"Found 'pass' in output from attempt {i+1}"
                                output = output_content
                                break

                return {
                    'success': False,
                    'error': str(e),
                    'output': output,
                    'execution_iteration': attempt + 1,
                    'translated_code': code,
                    'results_valid': is_valid,
                    'validation_message': validation_message,
                    'attempt_dir': run_attempt_dir,
                    'runtime_attempts_base_dir': runtime_attempts_base_dir
                }

        # Create run-specific directory for the final failed attempt
        run_attempt_dir = os.path.join(runtime_attempts_base_dir, f"run_{attempt}")
        os.makedirs(run_attempt_dir, exist_ok=True)

        # Save the final code to the run attempt directory
        # Use .cu extension for CUDA, .cpp for others
        pred_filename = f"pred_{attempt}.cu" if to_api == 'cuda' else f"pred_{attempt}.cpp"
        with open(os.path.join(run_attempt_dir, pred_filename), "w") as f:
            f.write(code)

        # Save the failure message to the run attempt directory
        with open(os.path.join(run_attempt_dir, "failure.txt"), "w") as f:
            f.write(f"Failed to run the code after {max_attempts} attempts")

        # Check if there's any output from previous attempts that might contain "pass"
        output = None
        is_valid = False
        validation_message = f'Failed to run the code after {max_attempts} attempts, could not verify results'

        for i in range(attempt):
            output_file = os.path.join(runtime_attempts_base_dir, f"run_{i}", "output.txt")
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    output_content = f.read()
                    if "pass" in output_content.lower():
                        # Found "pass" in a previous output
                        is_valid = True
                        validation_message = f"Found 'pass' in output from attempt {i+1}"
                        output = output_content
                        break

        return {
            'success': False,
            'error': f'Failed to run the code after {max_attempts} attempts',
            'output': output,
            'execution_iteration': max_attempts,
            'translated_code': code,
            'results_valid': is_valid,
            'validation_message': validation_message,
            'attempt_dir': run_attempt_dir,
            'runtime_attempts_base_dir': runtime_attempts_base_dir
        }

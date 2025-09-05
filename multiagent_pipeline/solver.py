import subprocess
import os
import logging
import shutil
from datetime import datetime
from utils import *
from backup_utils import check_for_pass, create_code_backup, create_error_backup, generate_result_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ExecutionAgent:
    def __init__(self, model_agent):
        self.model_agent = model_agent

    def copy_makefiles_and_additional_files(self, build_dir, kernel_name, to_api):
        """
        Copy Makefiles and create symlinks for all other files from hecbench to the compilation directory.
        This matches the implementation in eval_compile.py.

        Args:
            build_dir (str): The directory where compilation will be attempted
            kernel_name (str): The name of the kernel
            to_api (str): The target API (cuda, omp, etc.)
        """
        # Determine the source directory based on kernel name and API
        src_dir = None

        # Get the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Check if the kernel name contains a hyphen (e.g., "addBiasResidualLayerNorm-hip")
        if '-' in kernel_name:
            # The kernel name already includes the API, extract the base name
            base_kernel_name = kernel_name.split('-')[0]
            src_dir = os.path.join(project_root, 'multiagent_pipeline', 'hecbench_omp', 'src', f"{base_kernel_name}-{to_api}")
        else:
            # Try to find a matching directory
            src_dir = os.path.join(project_root, 'multiagent_pipeline', 'hecbench_omp', 'src', f"{kernel_name}-{to_api}")

        # Check if the source directory exists
        if not os.path.exists(src_dir):
            logging.warning(f"Source directory {src_dir} does not exist. Files will not be copied/symlinked.")
            return

        logging.info(f"Copying Makefiles and creating symlinks from {src_dir} to {build_dir}")

        changed_makefile_dirs = []  # Track directories with changed Makefiles

        # Process all files in the source directory
        for filename in os.listdir(src_dir):
            src_file = os.path.join(src_dir, filename)
            dst_file = os.path.join(build_dir, filename)

            # Remove existing file or symlink if it exists
            if os.path.exists(dst_file) or os.path.islink(dst_file):
                os.unlink(dst_file)

            # If it's a Makefile, copy it and modify paths
            if filename.lower().split('.')[0] == 'makefile' or filename.startswith("Makefile"):
                # Copy the file
                shutil.copy2(src_file, dst_file)

                # Modify paths in the Makefile to use absolute paths
                with open(dst_file, 'r') as f:
                    content = f.read()

                original_content = content

                # Replace relative paths with absolute paths
                import re
                modified_content = re.sub(
                    r'\.\./([^\s:="\']+)',
                    lambda m: str(os.path.join(src_dir, m.group(1))),
                    content
                )

                # Check if the content was actually modified
                if modified_content != original_content:
                    parent_dir = os.path.basename(build_dir)
                    changed_makefile_dirs.append(parent_dir)

                # Write the modified content back
                with open(dst_file, 'w') as f:
                    f.write(modified_content)

                logging.info(f"Copied and modified {src_file} to {dst_file}")
            else:
                # Skip creating symlinks for main.cpp, main.cu, and main.cpp: files
                if filename == "main.cpp" or filename == "main.cu" or filename == "main.cpp:" or filename == "main.cu:":
                    logging.info(f"Skipping symlink creation for {filename} as per requirements")
                else:
                    # For all other files, create a symlink
                    os.symlink(src_file, dst_file)
                    logging.info(f"Created symlink from {src_file} to {dst_file}")

        # Log the contents of the build directory
        logging.info(f"Contents of {build_dir} after copying files:")
        for filename in os.listdir(build_dir):
            logging.info(f"  {filename}")

        # Log directories with changed Makefiles
        if changed_makefile_dirs:
            logging.info(f"Modified Makefiles in the following directories: {', '.join(changed_makefile_dirs)}")

    @staticmethod
    def extract_source_filename_from_makefile(build_dir, to_api):
        """
        Extract the source filename from the Makefile in the build directory.

        Args:
            build_dir (str): The directory containing the Makefile
            to_api (str): The target API (cuda, omp, etc.)

        Returns:
            str: The source filename extracted from the Makefile, or a default value if not found
        """
        # Default filename based on API (fallback if extraction fails)
        default_filename = "main.cu" if to_api == 'cuda' else "main.cpp"

        # First check for a file named exactly "Makefile"
        makefile_path = os.path.join(build_dir, "Makefile")

        # If the standard Makefile doesn't exist, determine which API-specific Makefile to check
        if not os.path.exists(makefile_path):
            makefile_name = 'Makefile.aomp' if to_api == 'omp' else 'Makefile'
            if to_api == 'cuda':
                makefile_name = 'Makefile.nvc'

            makefile_path = os.path.join(build_dir, makefile_name)

            # Check if the API-specific Makefile exists
            if not os.path.exists(makefile_path):
                logging.warning(f"Makefile {makefile_path} not found. Using default filename: {default_filename}")
                return default_filename

        # Read the Makefile and look for the source line
        try:
            with open(makefile_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Look for lines like "source = main.cpp" or "SOURCE = main.cu"
                    if line.lower().startswith('source =') or line.lower().startswith('source='):
                        # Extract the filename
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            filename = parts[1].strip()
                            logging.info(f"Extracted source filename from Makefile: {filename}")
                            return filename

            # If we didn't find a source line, use the default
            logging.warning(f"Source line not found in Makefile. Using default filename: {default_filename}")
            return default_filename

        except Exception as e:
            logging.error(f"Error reading Makefile: {e}. Using default filename: {default_filename}")
            return default_filename

    def compile_and_run(self, build_dir, exec_name, kernel_name, max_attempts, from_api, to_api, initial_dir=None):
        attempt = 0
        codes = []
        errors = []
        code = None  # Initialize code variable

        # Define attempts_base_dir to avoid NameError
        attempts_base_dir = build_dir

        # Copy Makefiles and additional files to the compilation directory
        # self.copy_makefiles_and_additional_files(build_dir, kernel_name, to_api)

        # Determine the main file name by reading the Makefile
        main_filename = ExecutionAgent.extract_source_filename_from_makefile(build_dir, to_api)
        main_file_path = os.path.join(build_dir, main_filename)

        # For the first attempt, use the code from the initial directory
        if initial_dir and os.path.exists(initial_dir):
            # Find the pred_0 file in the initial directory
            pred_file = None
            for file in os.listdir(initial_dir):
                if file.startswith("pred_0") or file == exec_name:
                    pred_file = os.path.join(initial_dir, file)
                    break

            if pred_file and os.path.exists(pred_file):
                with open(pred_file, "r") as f:
                    code = f.read()

                # Make sure the main file has the code
                with open(main_file_path, "w") as f:
                    f.write(code)

                # Ensure we attempt at least one compilation when using an existing dataset
                # This ensures we try to compile at least once before moving to the model loop
                if max_attempts <= 0:
                    max_attempts = 1

        while attempt < max_attempts:
            try:
                # Determine the current filename
                if attempt == 0:
                    current_filename = f"pred_0.cu" if to_api == 'cuda' else f"pred_0.cpp"
                else:
                    # For subsequent attempts, we'll still track the attempt number but won't create separate files
                    current_filename = main_filename

                logging.info(f"Attempting to compile {current_filename} (Attempt {attempt + 1})")

                # For the first attempt, the code should already be in the main file
                # For subsequent attempts, we'll use the model to fix the code
                if attempt > 0:
                    # Update the main file with the new code from the model
                    with open(main_file_path, "w") as f:
                        f.write(code)

                # Read the current code from the main file
                with open(main_file_path, "r") as f:
                    code = f.read()

                # For the first attempt, save a copy as pred_0.cu/cpp
                if attempt == 0:
                    pred_0_path = os.path.join(build_dir, current_filename)
                    with open(pred_0_path, "w") as f:
                        f.write(code)

                    # Also save a copy in the kernel directory if initial_dir is provided
                    if initial_dir and os.path.exists(initial_dir):
                        create_code_backup(main_file_path, initial_dir, 'compile', 0)

                # env = os.environ.copy()  
                # env["LD_LIBRARY_PATH"] = f"{os.environ['CONDA_PREFIX']}/lib:" + env.get("LD_LIBRARY_PATH", "")

                # Log file and Makefile paths
                file_pwd = os.path.abspath(main_file_path)
                makefile_pwd = os.path.abspath(build_dir)
                output_dir = os.path.abspath(build_dir)

                logging.info(f"Compiling file: {file_pwd}")
                logging.info(f"Makefile directory: {makefile_pwd}")
                logging.info(f"Output directory: {output_dir}")

                try:
                    subprocess.run(['make', 'clean'], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except subprocess.CalledProcessError as e:
                    print(f"Make clean failed for {e.stderr}")
                
                if to_api == 'omp':
                    makefile_path = os.path.join(makefile_pwd, 'Makefile.aomp')
                    logging.info(f"Using Makefile: {makefile_path}")

                    # First try with DEVICE=cpu
                    logging.info("Attempting compilation with DEVICE=cpu")
                    compile_result = subprocess.run(
                        ['make', '-f', 'Makefile.aomp', 'DEVICE=cpu'],
                        cwd=build_dir,
                        capture_output=True,
                        text=True,
                        # env=env
                    )

                    # If that fails, try without specifying DEVICE
                    if compile_result.returncode != 0 and "unsupported option '-fopenmp'" in compile_result.stderr:
                        logging.info("Compilation with DEVICE=cpu failed due to unsupported OpenMP flags, trying without DEVICE")
                        compile_result = subprocess.run(
                            ['make', '-f', 'Makefile.aomp'],
                            cwd=build_dir,
                            capture_output=True,
                            text=True,
                            # env=env
                        )
                elif to_api == 'cuda':
                    makefile_path = os.path.join(makefile_pwd, 'Makefile')
                    logging.info(f"Using Makefile: {makefile_path}")
                    compile_result = subprocess.run(
                        ['make'],
                        cwd=build_dir,
                        capture_output=True,
                        text=True,
                        # env=env
                    )

                if compile_result.returncode != 0:
                    error_message = compile_result.stderr
                    errors.append(error_message)
                    codes.append(code)

                    # Save compilation error directly in the build directory
                    with open(os.path.join(build_dir, "compilation_error.txt"), "w") as f:
                        f.write(error_message)

                    logging.error(f"Compilation failed, trying to fix the code")

                    # Create context for the model agent
                    context = {
                        "kernel_name": kernel_name,
                        "from_api": from_api,
                        "to_api": to_api,
                        "attempt": attempt
                    }

                    # Get fixed code from the model
                    model_response = self.model_agent.fix_code(code, error_message, to_api, context=context)
                    if 'Error code: 400' in model_response:
                        break

                    # Extract code from the model response
                    fixed_code = extract_code_from_model_output(model_response)

                    # Update the main file with the fixed code
                    with open(main_file_path, "w") as f:
                        f.write(fixed_code)

                    # Save a copy in the kernel directory if initial_dir is provided
                    if initial_dir and os.path.exists(initial_dir):
                        # Create a backup of the fixed code with the pred_compile_X naming convention
                        create_code_backup(main_file_path, initial_dir, 'compile', attempt + 1)

                        # Create a backup of the error message
                        create_error_backup(error_message, initial_dir, 'compile', attempt)

                    # Prepare for next attempt
                    attempt += 1
                    code = fixed_code
                    continue

                # If compilation succeeded, the main file already has the correct code
                # No need to update it again

                # Clean up object files after successful compilation
                # try:
                #     logging.info(f"Cleaning up object files in {build_dir}")
                #     cleanup_result = subprocess.run(
                #         ['make', 'clean'],
                #         cwd=build_dir,
                #         capture_output=True,
                #         text=True,
                #         env=env
                #     )
                #     if cleanup_result.returncode == 0:
                #         logging.info("Object files cleaned up successfully")
                #     else:
                #         logging.warning(f"Failed to clean up object files: {cleanup_result.stderr}")
                # except Exception as e:
                #     logging.warning(f"Error during cleanup: {str(e)}")

                # Generate a partial result.txt file with compilation information
                # The run information will be added later by the RunnerAgent
                if initial_dir and os.path.exists(initial_dir):
                    try:
                        compile_attempt = 0 if attempt == 0 else attempt
                        generate_result_file(
                            initial_dir,
                            compile_attempt,
                            None,  # No compile error since compilation was successful
                            None,  # Run attempt will be filled in later
                            None,  # Run output will be filled in later
                            False  # Pass status will be determined later
                        )
                        logging.info(f"Generated partial result.txt file in {initial_dir}")
                    except Exception as e:
                        logging.warning(f"Error generating result.txt file: {str(e)}")

                return {
                    'success': True,
                    'output': compile_result.stdout,
                    'error': compile_result.stderr,
                    'compile_iteration': 0 if attempt == 0 else attempt + 1,  # Report 0 if compiled without model intervention, otherwise 1-indexed
                    'translated_code': code
                }
            except subprocess.TimeoutExpired:
                return {
                    'success': False,
                    'error': 'Process timed out',
                    'compile_iteration': 0 if attempt == 0 else attempt + 1,  # Report 0 if compiled without model intervention, otherwise 1-indexed
                    'translated_code': code
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'compile_iteration': 0 if attempt == 0 else attempt + 1,  # Report 0 if compiled without model intervention, otherwise 1-indexed
                    'translated_code': code
                }

        # Save the final code to the build directory
        with open(os.path.join(build_dir, main_filename), "w") as f:
            f.write(code)

        # Generate a result.txt file with compilation failure information
        if initial_dir and os.path.exists(initial_dir):
            try:
                # Get the last error message
                error_message = f'Failed to compile after {max_attempts} attempts'
                if errors:
                    error_message = errors[-1]

                generate_result_file(
                    initial_dir,
                    None,  # No successful compilation attempt
                    error_message,
                    None,  # No run attempt
                    None,  # No run output
                    False  # No pass status
                )
                logging.info(f"Generated result.txt file with compilation failure in {initial_dir}")
            except Exception as e:
                logging.warning(f"Error generating result.txt file: {str(e)}")

        return {
            'success': False,
            'error': f'Failed to compile after {max_attempts} attempts',
            'compile_iteration': 0 if attempt == 0 else attempt + 1,  # Report 0 if compiled without model intervention, otherwise 1-indexed
            'translated_code': code
        }

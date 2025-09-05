import os
import logging
import sys
from datetime import datetime

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(root_dir, 'data'))
from cuda_dataset import KernelDataset

HOME_PATH = os.path.expanduser('~')
# Use the actual project path instead of a hardcoded path
PROJECT_PATH = root_dir
DATASET_PATH = os.path.join(PROJECT_PATH, 'data', 'Datasets')
DATASET_NAME = 'llama3_8b_eval'

class QuestionerAgent:
    """
    Agent responsible for formulating translation requests for the model.
    Takes code in one API and requests its translation to another API.
    """
    def __init__(self, dataset_dir=None, model_agent=None, dataset_type='test', from_api=None, to_api=None):
        self.model_agent = model_agent

        # Initialize datasets if a directory is provided
        if dataset_dir:
            try:
                logging.info(f'Initializing dataset from {dataset_dir} with type {dataset_type}')
                # Use 'omp' as the parallel_api to ensure we get kernels with OpenMP API
                self.test_set = KernelDataset(dataset_dir, dataset_type=dataset_type, parallel_apis=['omp', 'cuda'], from_api=from_api, to_api=to_api)
                self.prompt_set = KernelDataset(dataset_dir, dataset_type='prompt', parallel_apis=['omp', 'cuda'])
                logging.info(f'Dataset loaded with size: {len(self.test_set)}')
            except Exception as e:
                logging.error(f'Error initializing dataset: {str(e)}')
                self.test_set = None
                self.prompt_set = None
        else:
            logging.info('No dataset directory provided')
            self.test_set = None
            self.prompt_set = None

    def build_message(self, from_api, to_api, code, num_shots=0):
        """Build a message for the model with few-shot examples if specified."""
        prompt_kernels = ['accuracy', 'gabor', 'permute']

        messages = [
            {"role": "system", "content": "You are an HPC expert specializing in translating between parallel programming APIs. Translate precisely and completely. Output only the translated code inside a code block. No explanations, no comments, no omissions."},
            {"role": "user", "content": f"For each kernel code provided, translate it from {from_api} to {to_api}. Provide the complete code in {to_api}. Do not truncate or use ellipses. Ensure correctness. Do not add any additional comments or explanations."},
            {"role": "assistant", "content": f"#Here is the translated code:\n#```{to_api}\n// Translated code goes here\n```"}
        ]

        # Add few-shot examples if requested
        if num_shots > 0 and self.prompt_set:
            for kernel in prompt_kernels[:num_shots]:
                try:
                    from_code = self.prompt_set.dataset[f"{kernel}-{from_api}"]
                    to_code = self.prompt_set.dataset[f"{kernel}-{to_api}"]

                    messages.append({"role": "user", "content": f'Translate the following code from {from_api} to {to_api}:\n{from_code}'})
                    messages.append({"role": "assistant", "content": f'Here is the translated code:\n{to_code}'})
                except KeyError:
                    logging.warning(f"Could not find example kernel {kernel} for {from_api}->{to_api}")
                    continue

        # Add the actual code to translate
        messages.append({"role": "user", "content": f'Translate the following code from {from_api} to {to_api}. Provide the complete code in {to_api}:\n{code}'})
        return messages

    def translate_code(self, from_api, to_api, code, num_shots=0, save_output=False, output_dir=None, kernel_name=None):
        """
        Translate code from one API to another using the model.

        Returns:
            str: The translated code, or None if the example should be skipped
        """
        if not self.model_agent:
            raise ValueError("ModelAgent must be provided to translate code")

        messages = self.build_message(from_api, to_api, code, num_shots=num_shots)

        # Create context for suspicious code detection
        context = {
            "kernel_name": kernel_name,
            "from_api": from_api,
            "to_api": to_api,
        }

        try:
            # Pass context to the model agent for suspicious code detection
            response = self.model_agent.generate_translation(messages, context=context)

            # If response is None, it means suspicious code was detected and the example should be skipped
            if response is None:
                logging.warning(f"Skipping example due to suspicious code detection")
                return None

            # Save outputs if requested
            if save_output and output_dir and response:
                os.makedirs(output_dir, exist_ok=True)

                with open(os.path.join(output_dir, 'source.cpp'), 'w') as f:
                    f.write(code)

                with open(os.path.join(output_dir, 'translated.cpp'), 'w') as f:
                    f.write(response)

                with open(os.path.join(output_dir, 'prompt.txt'), 'w') as f:
                    f.write("\n".join([msg["content"] for msg in messages]))

            return response

        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            return None

    def process_dataset(self, batch_size=1, num_shots=0, output_base_dir=None):
        """
        Process and translate all code samples in the dataset.

        If suspicious code is detected during translation, the example is flagged,
        saved for later review, and skipped.
        """
        if not self.test_set or not self.prompt_set:
            raise ValueError("Dataset not initialized")

        if not self.model_agent:
            raise ValueError("ModelAgent not provided")

        results = []
        skipped_count = 0

        for i in range(0, len(self.test_set), batch_size):
            batch = self.test_set[i:i+batch_size]

            for kernel in batch:
                kernel_name, from_api, from_code, to_api, to_code = kernel

                # Pass kernel_name to translate_code for better context
                translated_code = self.translate_code(
                    from_api, 
                    to_api, 
                    from_code, 
                    num_shots=num_shots,
                    kernel_name=kernel_name
                )

                # If translated_code is None, it means suspicious code was detected and the example should be skipped
                if translated_code is None:
                    logging.warning(f"Skipping {kernel_name}: {from_api}->{to_api} due to suspicious code detection")
                    skipped_count += 1
                    continue

                if translated_code and output_base_dir:
                    output_dir = os.path.join(output_base_dir, f"{kernel_name}_{from_api}_{to_api}")
                    os.makedirs(output_dir, exist_ok=True)

                    with open(os.path.join(output_dir, 'source.cpp'), 'w') as f:
                        f.write(from_code)

                    with open(os.path.join(output_dir, 'truth.cpp'), 'w') as f:
                        f.write(to_code)

                    with open(os.path.join(output_dir, 'translated.cpp'), 'w') as f:
                        f.write(translated_code)

                    logging.info(f"Processed {kernel_name}: {from_api}->{to_api}")

                results.append({
                    'kernel_name': kernel_name,
                    'from_api': from_api,
                    'to_api': to_api,
                    'original_code': from_code,
                    'translated_code': translated_code,
                    'ground_truth': to_code
                })

        if skipped_count > 0:
            logging.info(f"Skipped {skipped_count} examples due to suspicious code detection")

        return results


# Command-line interface preserved for backward compatibility
if __name__ == '__main__':
    import argparse
    from model_agent import ModelAgent

    parser = argparse.ArgumentParser(description="Few-Shot Learning Argument Parser")
    parser.add_argument('--num_shots', type=int, default=0, help='Number of shots for few-shot learning')
    parser.add_argument('--temp', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--max_token', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()

    # Create output directory
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        DATASET_PATH, 
        f'vllm_{DATASET_NAME}_shots={args.num_shots}_max_token={args.max_token}_temp={args.temp}_p={args.top_p}_{current_time}'
    )
    os.makedirs(output_path, exist_ok=True)

    # Set up logging to file in output directory
    log_file = os.path.join(output_path, "run.log")
    logging.basicConfig(
        filename=log_file, 
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Logging to {log_file}")

    # Load dataset
    dataset_dir = os.path.join(PROJECT_PATH, 'data/Datasets/HeCBench')

    # Initialize model agent
    model_agent = ModelAgent(
        model='meta-llama/Meta-Llama-3.1-8B-Instruct',
        max_tokens=args.max_token,
        temperature=args.temp,
        top_p=args.top_p
    )

    # Initialize questioner agent
    questioner = QuestionerAgent(dataset_dir=dataset_dir, model_agent=model_agent)

    # Process dataset
    questioner.process_dataset(
        batch_size=args.batch_size,
        num_shots=args.num_shots,
        output_base_dir=output_path
    )

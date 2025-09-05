from openai import OpenAI
import os
import logging
import argparse
import json
import sys
from datetime import datetime
from tqdm import tqdm

sys.path.append('data')#'../../data')
from kernel_dataset import KernelDataset

HOME_PATH = os.path.expanduser('~/unipar')#TODO: change to place with lots of storage!!
PROJECT_PATH = os.path.join(HOME_PATH, 'UniPar')#'SC-lab',

DATASET_PATH = os.path.join(HOME_PATH, 'Datasets')
DATASET_NAME = 'gpt-4o-mini_pcc'


def build_message(from_api, to_api, code, prompts, num_shots=0):
    prompt_kernels = ['accuracy', 'gabor', 'permute']

    messages = [
        {"role": "system", "content": "You are an HPC expert specializing in translating between parallel programming APIs."},
        {"role": "user", "content": f"For each kernel code provided, translate it from {from_api} to {to_api}. Provide the complete code in {to_api}. Do not truncate or use ellipses. Do not change the main function Ensure correctness. All function names must match."},
    ]
    
    for kernel in prompt_kernels[:num_shots]:
        from_code = prompts.dataset[f"{kernel}-{from_api}"]
        to_code = prompts.dataset[f"{kernel}-{to_api}"]

        messages.append({"role": "user", "content": f'Translate the following code  from {from_api} to {to_api}:\n{from_code}'})
        messages.append({"role": "assistant", "content": f' Here is the translated code:\n{to_code}'})

    messages.append({"role": "user", "content": f'Translate the following code  from {from_api} to {to_api}:\n{code}'})
    return messages


def translate_kernels(args, client, kernels, prompts, batch_size=4, num_return_sequences=3, output_dir=None):
    timer = 0
    for i in range(0, len(kernels), batch_size):
        batch = kernels[i:i+batch_size]

        batch_messages = [build_message(kernel[1], kernel[3], kernel[2], prompts, num_shots=args.num_shots) for kernel in batch]
        kernel = batch[0]
        kernel_name, from_api, from_code, to_api, to_code = kernel
        
        try:
            outputs_batch = client.chat.completions.create(
                model='gpt-4o-mini',
                # deployment_id = "gpt-4o-mini",
                messages=batch_messages[0],
                max_tokens=int(args.max_token),
                temperature=args.temp,
                top_p=args.top_p,
                n=num_return_sequences
            )
        except Exception as e:
            timer +=1
            logging.error(f"Error {kernel_name}: {from_api}->{to_api}, {timer}")
            logging.error(f"Error details: {e}")
            if timer > 50:
                logging.error("Too many errors. Exiting now.")
                exit(1)
            continue
        else:
            timer = 0

        # output_dir = str(output_dir)
        if output_dir is None:
            logging.error("Output directory not provided.")
            return
        output_path = os.path.join(output_dir, f'{kernel_name}_{from_api}_{to_api}')#TODO:check this path join path works?
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, 'msg.cpp'), 'w') as f:
            f.write("\n".join([msg["content"] for msg in batch_messages[0]]))

        with open(os.path.join(output_path, 'truth.cpp'), 'w') as f_truth:
            f_truth.write(to_code)

        for idx, output in enumerate(outputs_batch.choices):
            with open(os.path.join(output_path, f'pred_{idx}.cpp'), 'w') as f_pred:
                f_pred.write(output.message.content)

            logging.info(f"Processed {kernel_name}: {from_api}->{to_api}")

def get_directory_names(path):
    # Check if the path exists
    if not os.path.exists(path):
        return []
    return [item for item in os.listdir(path) 
            if os.path.isdir(os.path.join(path, item))]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Few-Shot Learning Argument Parser")
    parser.add_argument('--num_shots', type=int, required=True, help='Number of shots for few-shot learning')
    parser.add_argument('--temp', type=float, required=True)
    parser.add_argument('--top_p', type=float, required=True)
    parser.add_argument('--max_token', type=float, required=True)
    
    args = parser.parse_args()

    # Set up logging
    current_time = datetime.now().strftime("%d-%m-%y_%H-%M")
    log_file_name = f'gpt_pcc_few_shot={args.num_shots}_{current_time}.log'
    print("logfile_name:",log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load dataset
    dataset_dir = os.path.join(PROJECT_PATH, 'data/Datasets/HeCBench')
    # output_path = os.path.join(DATASET_PATH, f'llama3_70b_eval_shots={args.num_shots}_max_token={args.max_token}_temp={args.temp}_p={args.top_p}')
    out_path = os.path.join(DATASET_PATH, f'vllm_{DATASET_NAME}_shots={args.num_shots}_max_token={args.max_token}_temp={args.temp}_p={args.top_p}')
    to_ignore = get_directory_names(out_path)
    test_set = KernelDataset(dataset_dir, dataset_type='test_pcc',ignore_kernels=to_ignore)#, output_path=output_path)#######TODO:filter out already done in this run here
    logging.info(f'Dataset size: {len(test_set)}')

    # Set up few-shot prompts
    prompt_set = KernelDataset(dataset_dir, dataset_type='prompt_pcc')
    

    from openai import AzureOpenAI

    endpoint = ''
    model_name = "gpt-4o-mini"
    deployment = "gpt-4o-mini-2"

    subscription_key = ''
    api_version = "2025-01-01-preview"

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    
    translate_kernels(args, client, test_set, prompt_set, batch_size=1,num_return_sequences=1, output_dir=out_path)
    logging.info("Finished processing all kernels.")
    print("Finished processing all kernels.")

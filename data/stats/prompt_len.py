from transformers import AutoTokenizer
import json
import os
import itertools

HOME_PATH = os.path.expanduser('~')


def build_message(from_api, to_api, code, prompts, num_shots=0):
    prompt_kernels = ['accuracy', 'interval', 'medianfilter']

    messages = [
        {"role": "system", "content": "You are an HPC expert specializing in translating between parallel programming APIs."},
        {"role": "user", "content": f"For each code segment, please translate the following kernel written in {from_api} to {to_api}. Provide the full code in {to_api} without truncations or ellipses."},
    ]

    for kernel in prompt_kernels[:num_shots]:
        from_code = prompts[f"{kernel}-{from_api}"]
        to_code = prompts[f"{kernel}-{to_api}"]

        messages.append({"role": "user", "content": from_code})
        messages.append({"role": "assistant", "content": to_code})

    messages.append({"role": "user", "content": code})
    return messages


if __name__ == "__main__":
    NUM_SHOTS = 3
    APIS = ['cuda', 'omp', 'hip', 'sycl']

    # Set up few-shot prompt
    prompt_path = os.path.join(HOME_PATH, 'UniPar/data/Datasets/HeCBench/prompt.jsonl')
    prompts = []

    with open(prompt_path, 'r') as f:
        for line in f.readlines():
            prompts.append(json.loads(line.strip()))

    prompts = {f"{prompt['kernel_name']}-{prompt['parallel_api']}":prompt['code'] for prompt in prompts}

    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    for api1, api2 in itertools.permutations(APIS, 2):
        prompt = build_message(api1, api2, "", prompts, num_shots=NUM_SHOTS)

        tokens_amount = 0
        for message in prompt:
            tokens_amount += len(tokenizer.encode(message['content']))

        print(f"{api1}->{api2}: Prompt with {NUM_SHOTS} shots length {tokens_amount}")


# 0 shots
# cuda->omp: Prompt with 1 shots length 1746
# cuda->hip: Prompt with 1 shots length 1964
# cuda->sycl: Prompt with 1 shots length 2130
# omp->cuda: Prompt with 1 shots length 1746
# omp->hip: Prompt with 1 shots length 1759
# omp->sycl: Prompt with 1 shots length 1925
# hip->cuda: Prompt with 1 shots length 1964
# hip->omp: Prompt with 1 shots length 1759
# hip->sycl: Prompt with 1 shots length 2143
# sycl->cuda: Prompt with 1 shots length 2129
# sycl->omp: Prompt with 1 shots length 1924
# sycl->hip: Prompt with 1 shots length 2142

# 1 shots
# cuda->omp: Prompt with 0 shots length 47
# cuda->hip: Prompt with 0 shots length 47
# cuda->sycl: Prompt with 0 shots length 49
# omp->cuda: Prompt with 0 shots length 47
# omp->hip: Prompt with 0 shots length 47
# omp->sycl: Prompt with 0 shots length 49
# hip->cuda: Prompt with 0 shots length 47
# hip->omp: Prompt with 0 shots length 47
# hip->sycl: Prompt with 0 shots length 49
# sycl->cuda: Prompt with 0 shots length 48
# sycl->omp: Prompt with 0 shots length 48
# sycl->hip: Prompt with 0 shots length 48

# 3 shots
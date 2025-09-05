# UniPar -  Unified LLM-Based Framework for Parallel Code Translation in HPC

This project includes the code used in the [Unipar Paper](http:/link) 

<img margin: auto width="820" height="410" alt="Image" src="https://github.com/user-attachments/assets/79fb5340-7cfb-4a9a-aabc-2d58034d71a7" />

### System Overview
Multi-agent system for translating code between parallel programming APIs (e.g., CUDA to OpenMP) using language models with a feedback loop for error correction.

The system is comprised of:
- A pipeline for evaluating the LLaMA model (or model run with vllm)
- A similar pipeline for running GPT models using the API
- A multi-agent pipeline that can be run after the initial model is run
- A script for comparing compilation rate
- A script for running validation rate



The multi agent pipeline consists of three main components:

1. **QuestionerAgent**: Formulates translation requests for the model, including optional few-shot examples.
2. **ModelAgent**: Interfaces with the language model API to generate translations and fix code errors.
3. **ExecutionAgent**: Tests if the translated code compiles correctly, providing error feedback.

#### Feedback Loop Process

1. The QuestionerAgent sends the source code to the ModelAgent for translation
2. The ExecutionAgent attempts to compile and run the translated code
3. If compilation fails, the ExecutionAgent sends the error to the ModelAgent
4. The ModelAgent tries to fix the code based on the error
5. The cycle repeats until successful compilation or max iterations reached


### Usage

#### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Scientific-Computing-Lab/UniPar_AI.git
   cd UniPar
   ```

2. **Create and activate the conda environment**:
   ```bash
   conda env create -f env.yaml
   conda activate unipar
   ```

Ensure you have all the required dependencies listed in env.yaml before running the program.

**Note:** The HeCBench dataset is located in the multiagent pipeline folder.

## Example Scripts

### Full Run (`full_run.sh`)
Basic use of inference python script 
- When given the 'gpt' argument the script will run with the GPT inference otherwise it uses the standart inference.
- To use different models using the python llama inference script change the model parameter in the code (and the dataset name to label resulting folder correctly), 

Usage:
```bash
./full_run.sh [model_name]
```

### Full Evaluation Script (`full_evaluation_script.sh`)
This script  is an example of how to run all the nesesery scripts in order to convert LLM output into code, then compile and run it.
- Notice the parameters at the start of the that can be changed to evaluate different configurations.
- when using each script individually you can change the run_names parameter for result_analysis and eval_run to list multiple paths to run. 



Usage:
```bash
./full_evaluation_script.sh
```

### Full Run with Topic (`full_run_topic.sh`)
Executes the translation task with topic-specific configurations:
- Runs translation for all listed programming domains
- When given the 'gpt' argument the script will run with the GPT inference otherwise it uses the standart inference.

Usage:
```bash
./full_run_topic.sh [model_name]
```

### Full Run with Agent (`full_run_with_agent.sh`)
Executed the initial inference from the questioner model and then calls on the remaining pipline steps.
- this includes evaluation for the agent which is slightly diferent than the one for the base model.


Usage:
```bash
./full_run_with_agent.sh [dataset_type] [model_name] [max_iterations]
```




### Troubleshooting

#### Memory Issues
If you encounter memory issues:
1. Reduce `--max_tokens` to a lower value
2. Increase `--num_workers` if you have more GPUs available
3. Process smaller batches of the dataset at a time

#### Compilation Failures
If many kernels fail to compile:
1. Check the error messages in the compilation result files
2. Adjust the `--temperature` parameter (lower for more conservative translations)
3. Increase `--max_iterations` to allow more attempts at fixing compilation errors

#### Runtime Errors
If translated code compiles but fails at runtime:
1. Check the runtime error messages
2. Ensure the execution environment has the necessary libraries installed
3. Verify that the target hardware supports the target API (e.g., OpenMP)

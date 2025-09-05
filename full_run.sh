#!/bin/bash

source ~/.bashrc
source /home/tomerbitan/miniconda3/etc/profile.d/conda.sh

conda activate unipar

# echo "start with one"
if [[ $1 == "gpt" ]]; then
    SCRIPT="./llm_evaluation/new_gpt_inference.py"
else
    ./llm_evaluation/deploy_2.sh &

    while ! grep -q "Starting vLLM API server on http://" ./nohup.out; do
        sleep 10
    done
    SCRIPT="./llm_evaluation/llama_inference_2.py"
fi

python -u $SCRIPT --num_shots 0 --temp 0.2 --top_p 0.9 --max_token 15000
echo "done with one"
# python -u $SCRIPT --num_shots 1 --temp 0.2 --top_p 0.9 --max_token 15000 
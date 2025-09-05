#!/bin/bash

source ~/.bashrc
source /home/tomerbitan/miniconda3/etc/profile.d/conda.sh

conda activate unipar

# echo "start with one"
if [[ $1 == "gpt" ]]; then
    SCRIPT="./llm_evaluation/new_gpt_inference_topic.py"
else
    ./llm_evaluation/deploy_2.sh &

    while ! grep -q "Starting vLLM API server on http://" ./nohup.out; do
        sleep 10
    done
    SCRIPT="./llm_evaluation/llama_inference_2_topic.py"
fi

python -u $SCRIPT --num_shots 0 --temp 0.2 --top_p 0.9 --max_token 15000 --subgroup 'Bioinformatics'
echo 'done running with 0 shots'
python -u $SCRIPT --num_shots 1 --temp 0.2 --top_p 0.9 --max_token 15000 --subgroup 'Bioinformatics'

python -u $SCRIPT --num_shots 0 --temp 0.2 --top_p 0.9 --max_token 15000 --subgroup 'Computer_vision_and_image_processing'
echo 'done running with 0 shots'
python -u $SCRIPT --num_shots 1 --temp 0.2 --top_p 0.9 --max_token 15000 --subgroup 'Computer_vision_and_image_processing'

python -u $SCRIPT --num_shots 0 --temp 0.2 --top_p 0.9 --max_token 15000 --subgroup 'Data_compression_and_reduction'
echo 'done running with 0 shots'
python -u $SCRIPT --num_shots 1 --temp 0.2 --top_p 0.9 --max_token 15000 --subgroup 'Data_compression_and_reduction'

python -u $SCRIPT --num_shots 0 --temp 0.2 --top_p 0.9 --max_token 15000 --subgroup 'Data_encoding_decoding'
echo 'done running with 0 shots'
python -u $SCRIPT --num_shots 1 --temp 0.2 --top_p 0.9 --max_token 15000 --subgroup 'Data_encoding_decoding'

python -u $SCRIPT --num_shots 0 --temp 0.2 --top_p 0.9 --max_token 15000 --subgroup 'Language_and_kernel_features'
echo 'done running with 0 shots'
python -u $SCRIPT --num_shots 1 --temp 0.2 --top_p 0.9 --max_token 15000 --subgroup 'Language_and_kernel_features'

python -u $SCRIPT --num_shots 0 --temp 0.2 --top_p 0.9 --max_token 15000 --subgroup 'math'
echo 'done running with 0 shots'
python -u $SCRIPT --num_shots 1 --temp 0.2 --top_p 0.9 --max_token 15000 --subgroup 'math'

python -u $SCRIPT --num_shots 0 --temp 0.2 --top_p 0.9 --max_token 15000 --subgroup 'ml'
echo 'done running with 0 shots'
python -u $SCRIPT --num_shots 1 --temp 0.2 --top_p 0.9 --max_token 15000 --subgroup 'ml'

python -u $SCRIPT --num_shots 0 --temp 0.2 --top_p 0.9 --max_token 15000 --subgroup 'Simulation'
echo 'done running with 0 shots'
python -u $SCRIPT --num_shots 1 --temp 0.2 --top_p 0.9 --max_token 15000 --subgroup 'Simulation'
# python -u $SCRIPT --num_shots 1 --temp 0.2 --top_p 0.9 --max_token 15000 
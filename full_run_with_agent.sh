#!/bin/bash

source ~/.bashrc
source /home/tomerbitan/miniconda3/etc/profile.d/conda.sh

conda activate unipar

if [[ $1 == "gpt" ]]; then
    SCRIPT="./llm_evaluation/new_gpt_inference.py"
    ANSWER_NAME="gpt-4o-mini"
    OUT_ANSWER_NAME="gpt"
else
    ./llm_evaluation/deploy_2.sh &

    while ! grep -q "Starting vLLM API server on http://" ./nohup.out; do
        sleep 10
    done
    SCRIPT="./llm_evaluation/llama_inference_2.py"
    ANSWER_NAME="llama3.3_70b_eval"
    OUT_ANSWER_NAME="eval"
fi

NUM_SHOTS=0
TEMP=0.2
TOP_P=0.9
MAX_TOKEN=15000

python -u $SCRIPT --num_shots $NUM_SHOTS --temp $TEMP --top_p $TOP_P --max_token $MAX_TOKEN

echo "done with one"

conda activate unipar

OUTPATH="${OUT_ANSWER_NAME}_shot_${NUM_SHOTS}_m_${MAX_TOKEN}_t_${TEMP}_p_${TOP_P}"
python -u eval/compilation/eval_compile.py --model_out_files "vllm_${ANSWER_NAME}_shots=${NUM_SHOTS}_max_token=${MAX_TOKEN}_temp=${TEMP}_p=${TOP_P}" --extracted_code_paths $OUTPATH


python -u datasets_after_eval_compile_main.py --dataset_dir /home/tomerbitan/unipar/Datasets/eval/${OUTPATH} --from_api all --to_api omp >& "${OUTPATH}_agent_log_omp"

AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '{print $1}' | paste -sd "," -)
GPU_COUNT=$(echo $AVAILABLE_GPUS | tr ',' '\n' | wc -l)
if [ "$GPU_COUNT" -gt 0 ]; then
    #note that sometimes different GPU arcitectures behave slightly differently
    python -u datasets_after_eval_compile_main.py --dataset_dir /home/tomerbitan/unipar/Datasets/eval/${OUTPATH} --from_api all --to_api cuda >& "${OUTPATH}_agent_log_cuda"
fi

python parse_results.py --input_folder /home/tomerbitan/unipar/Datasets/eval/${OUTPATH} --summary out.csv
python multiagent_pipeline/get_data_from_summery.py --csv_file_path "out.csv"

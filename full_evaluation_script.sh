#!/bin/bash

source ~/.bashrc
source /home/tomerbitan/miniconda3/etc/profile.d/conda.sh

conda activate unipar

NUM_SHOTS=0
TEMP=0.2
TOP_P=0.9
MAX_TOKEN=15000
MODEL="gpt"

if [[ $MODEL == "gpt" ]]; then
    ANSWER_NAME="gpt-4o-mini"
    OUT_ANSWER_NAME="gpt"
else
    ANSWER_NAME="llama3.3_70b_eval"
    OUT_ANSWER_NAME="eval"
fi

OUTPATH="${OUT_ANSWER_NAME}_shot_${NUM_SHOTS}_m_${MAX_TOKEN}_t_${TEMP}_p_${TOP_P}"

python -u eval/compilation/eval_compile.py --model_out_files "vllm_${ANSWER_NAME}_shots=${NUM_SHOTS}_max_token=${MAX_TOKEN}_temp=${TEMP}_p=${TOP_P}" --extracted_code_paths $OUTPATH
python -u eval/compilation/utils/result_analysis_backup.py --run_names $OUTPATH
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '{print $1}' | paste -sd "," -)
GPU_COUNT=$(echo $AVAILABLE_GPUS | tr ',' '\n' | wc -l)
if [ "$GPU_COUNT" -gt 0 ]; then
    #note that sometimes different GPU arcitectures behave slightly differently
    python -u eval/run_and_speedup/eval_run.py --target 'cuda' --run_names $OUTPATH
fi
CORE_COUNT=$(nproc)
if [ "$CORE_COUNT" -gt 50 ]; then
    #note that we used 100 AMD EPYC 9334 changing this might warnet changing the timeout threshold
    python -u eval/run_and_speedup/eval_run.py --target 'omp' --run_names $OUTPATH
fi
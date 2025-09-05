#!/bin/bash
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '{print $1}' | paste -sd "," -)
export CUDA_VISIBLE_DEVICES=$AVAILABLE_GPUS
#0,1,2,3
# $AVAILABLE_GPUS
GPU_COUNT=$(echo $AVAILABLE_GPUS | tr ',' '\n' | wc -l)
#0,1
#,2,3
echo '################################ deploy_2 ##########################################' > nohup.out

echo "running vllm server" > ./vllm_server_status.txt

# nohup vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct\ #\-FP8
#neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8
    # --uvicorn-log-level debug \
    # --disable_custom_all_reduce \
# nohup vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct \
nohup vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --enable-chunked-prefill False \
    --tensor-parallel-size $GPU_COUNT \
    --max-model-len 80000 \
    --quantization="fp8" \
    --gpu-memory-utilization 0.8 \
    --swap-space 8 \
    --api-key token123 \
    --port 8001

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "crashed:$EXIT_CODE" >> ../vllm_server_status.txt
else
    echo "stopped:$EXIT_CODE" >> ../vllm_server_status.txt
fi


### prev version - transformers 4.48.2
### prev version - vllm 0.7.2

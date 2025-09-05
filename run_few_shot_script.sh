#!/bin/bash
while ! grep -q "Starting vLLM API server on http://" ./llm_evaluation/nohup.out; do
    sleep 10
done

python ./llm_evaluation/llama_inference_2_topic.py --num_shots 0 --temp 0.2 --top_p 0.9 --max_token 15000
echo "finished first"
python ./llm_evaluation/llama_inference_2_topic.py --num_shots 1 --temp 0.2 --top_p 0.9 --max_token 15000
echo "finished second"



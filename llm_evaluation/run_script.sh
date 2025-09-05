#!/bin/bash

python llm_evaluation/llama_inference_2.py --num_shots 0 --temp $1 --top_p $2 --max_token $3

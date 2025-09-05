#!/bin/bash

tune run --nproc_per_node 8 lora_finetune_distributed --config /home/unipar/UniPar/llm_finetune/configs/llama3.3_70B_lora.yaml \
	dataset.packed=False \
	compile=True \
	loss=torchtune.modules.loss.CEWithChunkedOutputLoss \
	enable_activation_checkpointing=True \
	optimizer_in_bwd=False \
	enable_activation_offloading=True \
	optimizer=torch.optim.AdamW \
	clip_grad_norm=1 \
	optimizer.lr=5e-6 \
	tokenizer.max_seq_len=16384 \
	gradient_accumulation_steps=4 \
	epochs=5 \
	batch_size=1

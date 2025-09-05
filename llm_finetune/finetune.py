import os
import logging
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, get_scheduler
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchtitan import ContextParallel

import sys
sys.path.append('../data')
from kernel_dataset import KernelDataset


HOME_PATH = os.path.expanduser('~')
PROJECT_PATH = os.path.join(HOME_PATH, 'SC-lab', 'UniPar')
MAX_LENGTH = 16_384
PRECISION = 2

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token




def tokenize_function(example):
    _, from_api, from_code, to_api, to_code = example

    system_message = {
        "role": "system",
        "content": "You are an HPC expert specializing in translating between parallel programming APIs."
    }
    user_message = {
        "role": "user",
        "content": f"Translate the following kernel from {from_api} to {to_api}. Provide the complete code in {to_api}. Do not truncate or use ellipses. Ensure correctness. \nCode:\n {from_code}"
    }
    assistant_message = {"role": "assistant", "content": to_code}

    chat_input = tokenizer.apply_chat_template(
        [system_message, user_message, assistant_message],
        tokenize=False,
        add_generation_prompt=False,
    )

    inputs = tokenizer(
        chat_input,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    # flatten from dict of 1-sample tensors to dict of tensors
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Remove batch dim

    if inputs['attention_mask'].sum().item() == MAX_LENGTH:
        return None

    # Mask out the prompt portion from the loss
    prompt_length = len(tokenizer.apply_chat_template(
        [system_message, user_message], tokenize=True
    ))
    labels = inputs['input_ids'].clone()
    labels[:prompt_length] = -100
    inputs['labels'] = labels

    return inputs


if __name__ == '__main__':

    logging.basicConfig(filename=f'peft_llama8.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        device_map = 'auto'
    )

    #  Apply LoRA to transformer modules
    lora_config = LoraConfig(
        r=8,        
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],  
        lora_dropout=0.1,  
    )

    num_gpus = torch.cuda.device_count()
    logging.info(f"Device count: {num_gpus}")

    # Integrating LoRA into the LLaMA model
    model = get_peft_model(model, lora_config)

    context_parallel = ContextParallel(model, num_gpus=num_gpus)
    model = context_parallel.wrap_model(model)


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(f"Total Parameters: {total_params}")
    logging.info(f"Trainable Parameters: {trainable_params}")
    logging.info(f"Memory Consumption: {(total_params + trainable_params) * PRECISION / 10**9} GB")

    # Prepare the dataset
    dataset_dir = os.path.join(PROJECT_PATH, 'data/Datasets/HeCBench')
    train_data = KernelDataset(dataset_dir, dataset_type='train')
    val_data = KernelDataset(dataset_dir, dataset_type='test')

    logging.info(f'Train size: {len(train_data)}')
    logging.info(f'Test size: {len(val_data)}')

    train_data = list(filter(lambda x: x is not None, [tokenize_function(sample) for sample in train_data]))
    val_data = list(filter(lambda x: x is not None, [tokenize_function(sample) for sample in val_data] ))
    
    logging.info(f'Filtered train size: {len(train_data)}')
    logging.info(f'Filtered test size: {len(val_data)}')

    # Create DataLoaders
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    eval_dataloader = DataLoader(val_data, batch_size=1)

    # Set up optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_train_epochs = 10
    num_training_steps = num_train_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Initialize Accelerator for distributed training and fp16 support
    accelerator = Accelerator(mixed_precision="fp16")
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Training loop
    global_step = 0
    for epoch in range(num_train_epochs):
        model.train()
        for batch in train_dataloader:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 10 == 0:
                logging.info(f"Epoch {epoch}, step {global_step}, loss: {loss.item()}")

        # Evaluation after each epoch
        model.eval()
        total_eval_loss = 0.0
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
            total_eval_loss += outputs.loss.item()
        avg_eval_loss = total_eval_loss / len(eval_dataloader)
        logging.info(f"Epoch {epoch} evaluation loss: {avg_eval_loss}")

    # Save the fine-tuned model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained("./ft_llama8")
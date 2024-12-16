import os 
import json
import pandas as pd
import torch
import gc
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from huggingface_hub import login

def clear_gpu_memory():
    """清理 GPU 記憶體"""
    gc.collect()
    torch.cuda.empty_cache()
    
# 設置使用第一張 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Hugging Face 登入
login(token="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")

# 量化配置
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_has_fp16_weight=False
)

# 讀取資料
with open('law/merged_laws.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 將資料轉換為列表格式
data_list = [{"law_name": k, "content": v["content"]} for k, v in data.items()]
df = pd.DataFrame(data_list)
ds = Dataset.from_pandas(df)
# ds = ds.select(range(16))
# print(ds[:3])

# 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained("yentinglin/Llama-3-Taiwan-8B-Instruct")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'

def find_latest_checkpoint(checkpoint_dir):
    """找到最新的checkpoint"""
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        return None
    # 提取數字並找到最大值
    checkpoint_nums = [int(cp.split('-')[-1]) for cp in checkpoints]
    latest_checkpoint = f"checkpoint-{max(checkpoint_nums)}"
    return os.path.join(checkpoint_dir, latest_checkpoint)

def process_func(example):
    MAX_LENGTH = 512
    prompt = f"{example['law_name']}: {example['content']}"
    # 修改為持續預訓練的格式
    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n你是一個專業的台灣律師，請記住並理解以下法條：{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
    # 讓模型重複法條內容作為response
    response = tokenizer(f"{example['law_name']}規定：{example['content']}<|eot_id|>", add_special_tokens=False)
    
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 處理資料集
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
print("input: ", tokenizer.decode(tokenized_id[0]['input_ids']))
print("output: ", tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[0]["labels"]))))

try:
    # 第一階段訓練
    print("開始第一階段訓練...")
    
    # 載入模型
    model = AutoModelForCausalLM.from_pretrained(
        "yentinglin/Llama-3-Taiwan-8B-Instruct", 
        device_map="auto", 
        quantization_config=quantization_config
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # 設置 LoRA
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 'gate_proj', 'up_proj', 'down_proj'],
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)

    # 訓練參數
    training_args = TrainingArguments(
        output_dir="llama-8b-continue-law",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=100,
        num_train_epochs=1,
        save_steps=100, # 为了快速演示，这里设置10，建议你设置成100
        learning_rate=3e-4,
        gradient_checkpointing=True,
        save_strategy="steps"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()
    
    # 釋放記憶體
    del trainer, model
    clear_gpu_memory()
    
    print("合併 LoRA 權重...")
    # 找到最新的checkpoint
    latest_checkpoint = find_latest_checkpoint("llama-8b-continue-law")
    if latest_checkpoint:
        print(f"使用最新的checkpoint: {latest_checkpoint}")


    # 合併 LoRA 權重
    base_model = AutoModelForCausalLM.from_pretrained(
        "yentinglin/Llama-3-Taiwan-8B-Instruct",
        device_map="auto",
        quantization_config=quantization_config
    )
    model = PeftModel.from_pretrained(base_model, latest_checkpoint)
    merged_model = model.merge_and_unload()
    # 保存合併後的模型
    merged_model.save_pretrained("merged_llama_8b_law_model")
    tokenizer.save_pretrained("merged_llama_8b_law_model")
    
    # 釋放記憶體
    del base_model, model, merged_model
    clear_gpu_memory()
    
    print("開始第二階段微調...")
    # 微調
    df_finetune = pd.read_json('law/train_data.jsonl', lines=True)
    ds_finetune = Dataset.from_pandas(df_finetune)
    # ds_finetune = ds_finetune.select(range(16))
    
    def process_func_finetune(example):
        MAX_LENGTH = 512
        text = example['title']
        if example['question'] is not None:
            text += example['question']
        instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n你是一個專業的台灣律師，請根據情境與問題回答相關法律，{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
        response = tokenizer(f"{example['label']}<|eot_id|>", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    tokenized_id_finetune = ds_finetune.map(process_func_finetune, remove_columns=ds_finetune.column_names)

    # 載入合併後的模型進行微調
    merged_model = AutoModelForCausalLM.from_pretrained(
        "merged_llama_8b_law_model",
        device_map="auto",
        quantization_config=quantization_config
    )
    merged_model.enable_input_require_grads()
    model_finetune = get_peft_model(merged_model, lora_config)
    

    training_args_finetune = TrainingArguments(
        output_dir="llama-8b-finetuned",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=100,
        num_train_epochs=50,
        save_steps=100, # 为了快速演示，这里设置10，建议你设置成100
        learning_rate=1e-4,
        gradient_checkpointing=True
    )

    trainer_finetune = Trainer(
        model=model_finetune,
        args=training_args_finetune,
        train_dataset=tokenized_id_finetune,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer_finetune.train()
    
    # 保存最終模型
    trainer_finetune.save_model("final_finetuned_model")
    
    # 釋放記憶體
    del merged_model, model_finetune, trainer_finetune
    clear_gpu_memory()

except Exception as e:
    print(f"訓練過程中發生錯誤: {str(e)}")
    # 確保發生錯誤時也清理記憶體
    clear_gpu_memory()

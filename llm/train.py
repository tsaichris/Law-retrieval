import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 設置使用第二張 GPU
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig, BitsAndBytesConfig, AutoModelForSequenceClassification
import torch
from peft import LoraConfig, TaskType, get_peft_model
from huggingface_hub import login, snapshot_download


login(token="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")

quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_has_fp16_weight=False
)

df = pd.read_json('law/train_data_related.jsonl', lines=True)
ds = Dataset.from_pandas(df)
print(ds[:3])
tokenizer = AutoTokenizer.from_pretrained("yentinglin/Llama-3-Taiwan-8B-Instruct")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'
'''
def process_func(example):
    MAX_LENGTH = 2048    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    text = example['title']
    if example['question'] is not None:
        text += example['question']
    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
    response = tokenizer(f"{example['label']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
'''

def process_func(example):
    MAX_LENGTH = 4096    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    text = example['title']
    if example['question'] is not None:
        text += example['question']
    # 合併 related_laws 為 list
    related_laws_text = ', '.join(example['related_laws'])
    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{text}\n以下是可能為答案的法條 : {related_laws_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
    response = tokenizer(f"{example['label']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
'''
def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n现在你要扮演皇帝身边的女人--甄嬛<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

'''



tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
print(tokenized_id)
print("input: ")
print(tokenizer.decode(tokenized_id[0]['input_ids']))
print("output: ")
print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[0]["labels"]))))

model = AutoModelForCausalLM.from_pretrained("yentinglin/Llama-3-Taiwan-8B-Instruct", device_map="auto", quantization_config=quantization_config)
model.config.pad_token_id = tokenizer.pad_token_id

model.enable_input_require_grads()
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 'gate_proj', 'up_proj', 'down_proj'],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=16, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
model = get_peft_model(model, config)
args = TrainingArguments(
    output_dir="llama-8b",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=100,
    num_train_epochs=75,
    save_steps=100, # 为了快速演示，这里设置10，建议你设置成100
    learning_rate=3e-4,
    gradient_checkpointing=True
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()



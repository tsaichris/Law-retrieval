import json
import csv
import argparse
from tqdm import tqdm
from peft import LoraConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_has_fp16_weight=False
    )
    return bnb_config

def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是一個專業的台灣律師，請根據情境與問題回答相關法律，{instruction}，回答:"

if __name__ == '__main__':
    device = 'cuda:0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',type=str,default="yentinglin/Llama-3-Taiwan-8B-Instruct")
    parser.add_argument('--peft_model',type=str,default="llama-8b/checkpoint-5400/")
    parser.add_argument('--input_json',type=str,default="law/test_data_related.jsonl")
    parser.add_argument('--output_csv',type=str,default="result.csv")
    args = parser.parse_args()
    bnb_config = get_bnb_config()
    config = LoraConfig.from_pretrained(args.peft_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb_config, device_map= device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, device_map= device)
    model = PeftModel.from_pretrained(model, args.peft_model, device_map= device)
    
    with open(args.input_json, "r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f]



    result_list = []
    count = 0
    
    for i in tqdm(range(len(data))):
        result_dict = {}
        text = data[i]['title']
        if data[i]['question'] is not None:
            text += data[i]['question']
            
        related_laws_text = ', '.join(data[i]['related_laws'])
        text += "以下是可能為答案的法條 : " + related_laws_text
        
        prompt = get_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=64)
        result_dict['id'] = data[i]['id']
        result_dict['output'] = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
        print(result_dict)
        result_list.append(result_dict)

    with open(args.output_csv, "w", encoding='utf-8', newline='') as csvfile:
        fieldnames = ['id', 'TARGET']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in result_list:
            writer.writerow({
                'id': result['id'],
                'TARGET': result['output']
            })

    print(f"Results have been saved to {args.output_csv}")
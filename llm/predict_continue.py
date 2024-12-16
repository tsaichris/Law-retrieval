import json
import csv
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_has_fp16_weight=False
    )

def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"<|start_header_id|>user<|end_header_id|>\n\n你是一個專業的台灣律師，請根據情境與問題回答相關法律，{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

if __name__ == '__main__':
    device = 'cuda:0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetuned_model', type=str, default="final_finetuned_model/")
    parser.add_argument('--input_json', type=str, default="test_data.jsonl")
    parser.add_argument('--output_csv', type=str, default="result.csv")
    args = parser.parse_args()
    
    bnb_config = get_bnb_config()
    
    model = AutoModelForCausalLM.from_pretrained(
        args.finetuned_model,
        quantization_config=bnb_config,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.finetuned_model, device_map=device)

    with open(args.input_json, "r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    result_list = []
    for i in tqdm(range(len(data))):
        result_dict = {}
        text = data[i]['title']
        if data[i]['question'] is not None:
            text += data[i]['question']
        prompt = get_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=128, num_beams=20)
        result_dict['id'] = data[i]['id']
        result_dict['output'] = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
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
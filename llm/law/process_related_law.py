import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import re
from tqdm import tqdm
from torch import cosine_similarity
def load_model():
    model_id = "yentinglin/Llama-3-Taiwan-8B-Instruct" #lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct
    quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_has_fp16_weight=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, device_map= 'cuda:1')
    model = AutoModel.from_pretrained(model_id, quantization_config=quantization_config, device_map= 'cuda:1')
    # model = AutoModel.from_pretrained(model_id).to('cuda:0')
    model.eval()
    return tokenizer, model

def generate_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to('cuda:1') for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
    return sentence_embedding.cpu()

def find_similar_laws(query_embedding, laws_data, top_k=25):
    similarities = {}
    query_embedding = torch.tensor(query_embedding)
    
    for law_id, law_data in laws_data.items():
        law_embedding = torch.tensor(law_data['embedding'])
        similarity = cosine_similarity(query_embedding.unsqueeze(0), 
                                    law_embedding.unsqueeze(0)).item()
        similarities[law_id] = similarity
    
    # 排序並取得前25個最相似的法條
    sorted_laws = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [law_id for law_id, _ in sorted_laws]

def process_jsonl():
    # 載入模型
    tokenizer, model = load_model()
    
    # 讀取法條資料
    with open('merged_laws.json', 'r', encoding='utf-8') as f:
        laws_data = json.load(f)
    
    # 讀取並處理第一個JSONL條目
    jsonl_entry = {
        "id": 0,
        "title": "帶手指虎打傷人，手指虎會被「沒收」還是「沒入」？",
        "question": "知道沒收是刑法，沒入是行政法。\n\r\n單純持有違禁品（手指虎）會遭到沒收，\r\n但用違禁品傷人，是會被「沒收」還是「沒入」呢？"
    }
    
    # 生成查詢的embedding
    combined_text = f"{jsonl_entry['title']} {jsonl_entry['question']}"
    query_embedding = generate_embedding(combined_text, tokenizer, model)
    
    # 找出最相似的法條
    similar_laws = find_similar_laws(query_embedding, laws_data)
    
    # 更新JSONL條目
    jsonl_entry['similar_laws'] = similar_laws
    
    # 儲存結果
    with open('output.jsonl', 'w', encoding='utf-8') as f:
        f.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')
    
    print("處理完成，結果已儲存至 output.jsonl")
    print("\n最相似的前5個法條：")
    for law in similar_laws[:5]:
        print(law)

if __name__ == "__main__":
    process_jsonl()
'''
def generate_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
    return sentence_embedding.cpu().numpy()

def extract_law_name(law_id):
    match = re.match(r'(.+?)第\d+(?:之\d+)?條', law_id)
    if match:
        return match.group(1)
    return law_id

def find_similar_laws(query_embedding, laws_data, top_k=1000):
    similarities = {}
    query_embedding = np.array(query_embedding)
    
    for law_id, law_data in laws_data.items():
        law_embedding = np.array(law_data['embedding'])
        similarity = -np.linalg.norm(query_embedding - law_embedding)
        similarities[law_id] = similarity
    
    sorted_laws = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    law_names = set(extract_law_name(law_id) for law_id, _ in sorted_laws)
    return list(law_names)

def calculate_coverage(label_laws, predicted_laws):
    label_set = set(label_laws)
    predicted_set = set(predicted_laws)
    
    matched_laws = label_set.intersection(predicted_set)
    coverage = len(matched_laws) / len(label_set) if label_set else 0
    
    return coverage, matched_laws, label_set

def count_characters(text):
    """計算字串中的字元數"""
    return len(text)

def process_data_file(input_file, output_file, tokenizer, model, laws_data, calculate_coverage_flag=False):
    print(f"處理 {input_file}...")
    
    entries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line))
    
    total_coverage = 0
    total_entries = 0
    max_char_count = 0

    for entry in tqdm(entries, desc=f"處理 {input_file}"):
        combined_text = f"{entry['title']} {entry['question']}"
        query_embedding = generate_embedding(combined_text, tokenizer, model)
        similar_laws = find_similar_laws(query_embedding, laws_data)
        
        # 添加相關法規
        entry['related_laws'] = similar_laws
        
        # 計算字數
        related_laws_text = ', '.join(similar_laws)
        total_text = f"{related_laws_text} {entry['question']} {entry['title']}"
        char_count = count_characters(total_text)
        entry['char_count'] = char_count
        print(char_count)
        # 更新最大字數
        max_char_count = max(max_char_count, char_count)
        
        # 如果有label欄位且需要計算coverage
        if calculate_coverage_flag and 'label' in entry:
            label_laws = set(extract_law_name(law) for law in entry['label'].split(','))
            coverage, matched_laws, _ = calculate_coverage(label_laws, similar_laws)
            entry['coverage'] = coverage
            entry['matched_laws'] = list(matched_laws)
            print(coverage)
            total_coverage += coverage
            total_entries += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    average_coverage = total_coverage / total_entries if total_entries > 0 else 0
    print(f"已完成處理並儲存至 {output_file}")
    print(f"平均覆蓋率: {average_coverage * 100:.2f}%")
    print(f"最大字數: {max_char_count}")

def main():
    tokenizer, model = load_model()
    with open('merged_laws.json', 'r', encoding='utf-8') as f:
        laws_data = json.load(f)
    
    # 處理訓練資料（包含coverage計算）
    process_data_file('train_data.jsonl', 'train_data_related.jsonl', 
                     tokenizer, model, laws_data, calculate_coverage_flag=True)
    
    # 處理測試資料（不計算coverage）
    process_data_file('test_data.jsonl', 'test_data_related.jsonl', 
                     tokenizer, model, laws_data, calculate_coverage_flag=False)

if __name__ == "__main__":
    main()
'''
import json

def process_jsonl():
    processed_data = []

    # 讀取 train_data.jsonl 文件
    with open('train_data.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            
            # 移除 "label" 欄位
            if 'label' in entry:
                del entry['label']
            
            processed_data.append(entry)

    # 將處理後的數據保存為 train_data_processed.json
    with open('train_data_processed.json', 'w', encoding='utf-8') as outfile:
        json.dump(processed_data, outfile, ensure_ascii=False, indent=2)

    print("處理完成。結果已保存至 train_data_processed.json")

if __name__ == "__main__":
    process_jsonl()

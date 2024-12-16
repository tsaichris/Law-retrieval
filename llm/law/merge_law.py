import os
import json

def merge_json_files():
    # 取得當前目錄下所有的 JSON 檔案
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    
    # 建立一個字典來存放所有資料
    merged_data = {}
    
    # 讀取每個 JSON 檔案並合併
    for json_file in json_files:
        try:
            print(f"正在處理: {json_file}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 移除 'embedding' 欄位
                for key, value in data.items():
                    if isinstance(value, dict) and 'embedding' in value:
                        del value['embedding']
                merged_data.update(data)
        except Exception as e:
            print(f"處理 {json_file} 時發生錯誤: {str(e)}")
    
    # 將合併後的資料寫入新的 JSON 檔案
    output_file = 'merged_laws.json'
    print(merged_data)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    
    print(f"合併完成，已儲存至 {output_file}")

if __name__ == "__main__":
    merge_json_files()
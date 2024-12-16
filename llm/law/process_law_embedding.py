import os
import json
import torch
from openai import OpenAI

# 設定 Perplexity API
os.environ["PERPLEXITY_API_KEY"] = "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"
client = OpenAI(
    api_key=os.environ["PERPLEXITY_API_KEY"],
    base_url="https://api.perplexity.ai"
)

def get_perplexity_embedding(text):
    text = text.replace("\n", " ")
    # 使用 Perplexity 的模型
    response = client.embeddings.create(
        model="llama-3.1-sonar-small-128k-online",  
        input=text
    )
    return response.data[0].embedding

def generate_embedding(key, value):
    combined_text = f"{key} {value}"
    return get_perplexity_embedding(combined_text)

def process_json_files():
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    
    for json_file in json_files:
        try:
            print(f"處理檔案: {json_file}")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for key, value in data.items():
                if isinstance(value, dict) and 'content' in value:
                    content = value['content']
                else:
                    content = value
                
                embedding = generate_embedding(key, content)
                
                data[key] = {
                    "content": content,
                    "embedding": embedding
                }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                
            print(f"已完成 {json_file} 的embedding處理")
            
        except Exception as e:
            print(f"處理 {json_file} 時發生錯誤: {str(e)}")

def main():
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用，請確認GPU設置")
            
        process_json_files()
    except Exception as e:
        print(f"程式執行錯誤: {str(e)}")

if __name__ == "__main__":
    main()
import json

class LawReader:
    def __init__(self, json_file='merged_laws.json'):
        """初始化並讀取JSON檔案"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                self.laws = json.load(f)
            print(f"成功載入法條資料，共 {len(self.laws)} 條")
        except Exception as e:
            print(f"讀取檔案時發生錯誤: {str(e)}")
            self.laws = {}

    def get_law_by_id(self, law_id):
        """根據法條編號查詢"""
        if law_id in self.laws:
            return self.laws[law_id]
        return None

    def search_by_keyword(self, keyword):
        """關鍵字搜尋（搜尋法條內容）"""
        results = {}
        for law_id, law_data in self.laws.items():
            if 'content' in law_data and keyword in law_data['content']:
                results[law_id] = law_data
        return results

    def get_law_content(self, law_id):
        """只取得法條內容"""
        law = self.get_law_by_id(law_id)
        if law and 'content' in law:
            return law['content']
        return None

    def get_law_embedding(self, law_id):
        """取得法條的 embedding"""
        law = self.get_law_by_id(law_id)
        if law and 'embedding' in law:
            return law['embedding']
        return None

    def print_law_info(self, law_id):
        """印出完整法條資訊"""
        law = self.get_law_by_id(law_id)
        if law:
            print(f"法條編號: {law_id}")
            print(f"內容: {law.get('content', 'N/A')}")
            print(f"Embedding 長度: {len(law.get('embedding', []))}")
        else:
            print(f"找不到法條: {law_id}")

def main():
    # 建立讀取器實例
    reader = LawReader()

    # 示範使用方式
    print("\n=== 使用範例 ===")
    
    # 1. 直接查詢特定法條
    print("\n1. 查詢特定法條:")
    law_id = "高速公路及快速公路交通管制規則第1條"  # 請替換成實際的法條編號
    reader.print_law_info(law_id)

    # 2. 關鍵字搜尋
    print("\n2. 關鍵字搜尋:")
    keyword = "歧視"  # 請替換成想要搜尋的關鍵字
    results = reader.search_by_keyword(keyword)
    print(f"包含關鍵字 '{keyword}' 的法條數量: {len(results)}")
    for law_id in list(results.keys())[:3]:  # 只顯示前3筆結果
        print(f"\n{law_id}:")
        print(reader.get_law_content(law_id))

    # 3. 取得 embedding
    print("\n3. 取得法條 embedding:")
    embedding = reader.get_law_embedding(law_id)
    if embedding:
        print(f"Embedding 維度: {len(embedding)}")

if __name__ == "__main__":
    main()
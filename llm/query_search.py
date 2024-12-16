import json
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm





class QueryRewriter:
    def __init__(self, model_name="yentinglin/Llama-3-Taiwan-8B-Instruct"):
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="float16"
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map= 'auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.prompt_template = """你是一個專業的法律查詢重寫助手。請將以下查詢重寫成更精確的法律查詢語句，以便更容易檢索到相關法律條文：
原始查詢: {title} {question}。重寫後的查詢："""

    def rewrite_query(self, title, question):
        prompt = self.prompt_template.format(
            title=title,
            question=question
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
        print(response)
        return response
class DocumentRelevanceModel:
    def __init__(self, model_name="BAAI/bge-large-zh-v1.5", reranker_name="BAAI/bge-reranker-large"):
        # 初始化 embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # 初始化 reranker model
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_name)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_name)
        self.reranker_model = self.reranker_model.to(self.device)
        self.reranker_model.eval()
        
        self.documents = {}
        self.doc_embeddings = None
        
    def get_embedding(self, text):
        text = f"query: {text}" if "query: " not in text else text
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()
    
    def load_documents(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
            
        self.doc_contents = []
        self.doc_ids = []
        
        print("開始處理文件...")
        for doc_id, doc_info in self.documents.items():
            combined_text = f"{doc_id} {doc_info['content']}"
            self.doc_contents.append(combined_text)
            self.doc_ids.append(doc_id)
        
        print("正在生成文件嵌入向量...")
        self.doc_embeddings = np.vstack([
            self.get_embedding(content) for content in tqdm(self.doc_contents)
        ])
        

        print("文件處理完成！")
        
    def rerank_documents(self, query, documents, top_k=10):
        
        pairs = [[query, doc['combine_text']] for doc in documents]
        print(documents[0]['combine_text'])
        with torch.no_grad():
            inputs = self.reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.device)
            
            scores = self.reranker_model(**inputs).logits.view(-1,).float()
            scores = scores.cpu().numpy()
        
        # 將重排序分數加入文件中
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # 根據重排序分數排序
        reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return reranked_docs[:top_k]
    
    def find_relevant_documents(self, query, initial_top_k=100, final_top_k=25):
        # 第一階段：使用 embedding 檢索
        query_embedding = self.get_embedding(query)
        similarities = cosine_similarity(query_embedding, self.doc_embeddings).flatten()
        top_indices = similarities.argsort()[-initial_top_k:][::-1]
        
        initial_results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                doc_id = self.doc_ids[idx]
                content = self.documents[self.doc_ids[idx]]['content']
                initial_results.append({
                    'doc_id': self.doc_ids[idx],
                    'content': self.documents[self.doc_ids[idx]]['content'],
                    'combine_text' : f"{doc_id} {content}",
                    'similarity': float(similarities[idx])
                })
        
        # 第二階段：使用 reranker 重新排序
        if len(initial_results) > 0:
            reranked_results = self.rerank_documents(query, initial_results, final_top_k)
            return reranked_results
        return initial_results


def calculate_coverage(relevant_docs, true_labels):
    # 將相關文件ID轉換為集合
    predicted_labels = set([doc['doc_id'] for doc in relevant_docs])
    true_labels = set(true_labels.split(','))
    
    # 計算涵蓋率
    covered_labels = predicted_labels.intersection(true_labels)
    coverage = len(covered_labels) / len(true_labels) if true_labels else 0
    
    return coverage, list(covered_labels)

def main():
    print("初始化模型中...")
    model = DocumentRelevanceModel()
    query_rewriter = QueryRewriter()
    # rewritten_query = query_rewriter.rewrite_query("雇用工讀生", "公司辦展覽欲雇用工讀生看顧器材，分為兩個時段各四個小時\n\r\nQ1, 若工讀生自願連續工作八個小時不休息，這樣是否違反勞基法？ 那若雙方協議可以嗎？ 如果不行，休息時間是否需要照算薪水？（例如來上班八個小時，安排休息一個小時，薪水是算八小時還是七小時）\n\r\nQ2, 工讀生若在國定假日上班，薪水也是否給雙倍？\n\r\nQ3, 如果要跟工讀生訂定工作契約來確保工讀生須對設備保管責任，哪類相關契約可以參考？")
    
    
    print("載入法條文件...")
    model.load_documents('law/merged_laws.json')
    
    # 讀取訓練數據
    train_data = []
    
    with open('train_data.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    # 處理每個訓練樣本
    results = []
    print("處理訓練數據...")
    for sample in tqdm(train_data):
        # combined_text = f"{sample['title']} {sample['question']}"
        rewritten_query = query_rewriter.rewrite_query(sample['title'], sample['question'])
        
        # 搜尋相關文件
        relevant_docs = model.find_relevant_documents(rewritten_query)
        
        # 計算涵蓋率
        coverage, covered_labels = calculate_coverage(relevant_docs, sample['label'])
        
        # 儲存結果
        result = {
            'id': sample['id'],
            'title': sample['title'],
            'question': sample['question'],
            'true_labels': sample['label'].split(','),
            'relevant_docs': relevant_docs,
            'coverage': coverage,
            'covered_labels': list(covered_labels)
        }
        results.append(result)
    
    # 儲存結果
    print("儲存結果...")
    with open('law/train_data_related.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 計算平均涵蓋率
    avg_coverage = sum(r['coverage'] for r in results) / len(results)
    print(f"\n平均標籤涵蓋率: {avg_coverage:.4f}")
    
if __name__ == "__main__":
    main()

import chromadb
from transformers import AutoTokenizer, AutoModel
import torch
from chromadb.utils.embedding_functions import EmbeddingFunction
from huggingface_hub import login
import logging

class CustomLlamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        # Authenticate with Hugging Face
        login(token="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")
        
        self.model_id = "lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.eval()
        """
        self.instruction = 
        將日常問題轉換為可以匹配法律條文的表示方式。
        需要考慮：
        1. 將一般民眾的日常用語對應到專業法律用語
        2. 識別實際案例中描述的情境和法條關聯
        3. 保留問題的核心概念和關鍵要素
        4. 將問題與相似的法律概念和情境連結
        請轉換以下民眾提問：
        """
        self.instruction = ''

    
    def __call__(self, input: list[str]) -> list[list[float]]:
        formatted_inputs = [f'{self.instruction}\n{text}' for text in input]
                    
        print("\nFormatted query with instruction:")
        print(formatted_inputs)
        all_embeddings = []
        for text in formatted_inputs:
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                outputs = self.model(**inputs)
                
                embeddings = outputs.last_hidden_state
                sentence_embedding = torch.mean(embeddings, dim=1).squeeze()
                
                embedding_list = sentence_embedding.tolist()
                norm = sum(x*x for x in embedding_list) ** 0.5
                normalized_embedding = [x/norm for x in embedding_list]
                
                all_embeddings.append(normalized_embedding)
        
        return all_embeddings

class LawSearcher_llama:
    def __init__(self, client_path:str = "vectorDB/law_original_llama1"):
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler('logs/searcher_result.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)

        self.logger.info("Initializing Llama model and ChromaDB client...")
    
        self.embedding_function = CustomLlamaEmbeddingFunction()
        # Use the same database path and collection name as in the first code
        self.client = chromadb.PersistentClient(client_path)
        self.collection = self.client.get_collection(
            name="law_collection_llama_em1",
            embedding_function=self.embedding_function
        )

    def search(self, query: str, n_results: int = 25):
        try:
            self.logger.info("\nProcessing search query...")
            self.logger.info(f"Query: {query}")
            
            
            # Use the Llama model's embedding function directly
            query_embedding = self.embedding_function([query])[0]
            
            self.logger.info(f"\nQuery embedding length: {len(query_embedding)}")
            self.logger.info(f"Sample of query embedding (first 5 values): {query_embedding[:5]}")
            
            res = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['distances', 'embeddings', 'documents', 'metadatas']
            )
            
            self.logger.info("\nQuery Results:", "\n--------------")
            
            results = []
            for i in range(len(res['documents'][0])):
                result = {
                    "rank": i + 1,
                    "file": res['metadatas'][0][i]['file_name'],
                    "index": res['metadatas'][0][i]['original_index'],
                    "text": res['documents'][0][i],
                    "distance": res['distances'][0][i]
                }
                results.append(result)
                self.logger.info(f"\nRank {i+1}:")
                self.logger.info(f"File: {result['file']}")
                self.logger.info(f"Index: {result['index']}")
                self.logger.info(f"Text: {result['text']}")
                self.logger.info(f"Distance: {result['distance']}")
            
            return results
            
        except Exception as e:
            self.logger.info(f"Vector search failed: {e}")
            return []

def main():
    searcher = LawSearcher_llama ()
    
    # Example query
    query = '帶手指虎打傷人，手指虎會被「沒收」還是「沒入」。知道沒收是刑法，沒入是行政法。單純持有違禁品（手指虎）會遭到沒收，但用違禁品傷人，是會被「沒收」還是「沒入」呢？'
    
    searcher.search(query)

if __name__ == "__main__":
    main()
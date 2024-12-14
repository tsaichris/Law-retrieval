import chromadb
from transformers import AutoTokenizer, AutoModel
import torch
import os
from chromadb.utils.embedding_functions import EmbeddingFunction
from huggingface_hub import login
import numpy as np

# Authenticate with Hugging Face
login(token="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")

class CustomLlamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model_id = "lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.eval()  # Set model to evaluation mode
        
        # Chunking parameters
        self.chunk_size = 512
        self.stride = 256
        self.length_threshold = 512  # When to start chunking

        # Similar instruction concept from the original code but adapted for the context
        self.instruction = """
        將法律條文轉換為可以匹配日常問題的表示方式。
        需要考慮：
        1. 將專業法律用語對應到一般民眾的日常用語
        2. 識別法條中描述的情境和實際案例的關聯
        3. 保留法條的核心概念和關鍵要素
        4. 連結相似的法律概念和情境
        請轉換以下法條：
        """
    def _get_chunks(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        formatted_text = f"{self.instruction}\n{text}"
        tokens = self.tokenizer.encode(formatted_text)
        
        if len(tokens) <= self.length_threshold:
            return [formatted_text]
            
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start += self.stride
        
        return chunks
    
    def _embed_single(self, text: str) -> list[float]:
        """Embed a single piece of text."""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                  max_length=self.chunk_size)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            sentence_embedding = torch.mean(embeddings, dim=1).squeeze()
            return sentence_embedding.tolist()
    
    def __call__(self, input_texts: list[str]) -> list[list[float]]:
        """Process a batch of texts."""
        all_embeddings = []
        secondary_batch_size = 3  # Process 3 chunks at a time for long documents
        
        for text in input_texts:
            chunks = self._get_chunks(text)
            
            if len(chunks) == 1:
                # Short document - process normally
                embedding = self._embed_single(chunks[0])
                # Normalize the embedding
                norm = sum(x*x for x in embedding) ** 0.5
                normalized_embedding = [x/norm for x in embedding]
                all_embeddings.append(normalized_embedding)
            else:
                # Long document - process chunks in small batches
                chunk_embeddings = []
                for i in range(0, len(chunks), secondary_batch_size):
                    batch_chunks = chunks[i:i + secondary_batch_size]
                    batch_embeddings = [self._embed_single(chunk) for chunk in batch_chunks]
                    chunk_embeddings.extend(batch_embeddings)
                
                # Average all chunk embeddings
                avg_embedding = np.mean(chunk_embeddings, axis=0)
                # Normalize the final embedding
                norm = np.sqrt(np.sum(avg_embedding ** 2))
                normalized_embedding = (avg_embedding / norm).tolist()
                all_embeddings.append(normalized_embedding)
        
        return all_embeddings


def parse_line(line: str) -> tuple:
    """Parse a single line into index and text."""
    parts = line.strip().split(' ', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None

def getDataFromText(directory_path: str, summary) -> list:
    """Read data from section.txt in the directory."""
    if summary:
        result = []
    
        # Read the file and process each line
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Split the line by comma
                parts = line.strip().split(',')
                
                # Extract the law name (remove '法規名稱：' prefix)
                law_name = parts[0].replace('法規名稱：', '').strip()
                
                # Add to result list [law_name, original_content]
                result.append((law_name, line.strip()))
        
        return result
    
    else:

        """Read data from multiple text files in the directory."""
        all_data = []
        try:
            print(f"Reading files from directory: {directory_path}")
            for filename in os.listdir(directory_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(directory_path, filename)
                    print(f"Processing file: {filename}")
                    
                    with open(file_path, 'r', encoding="utf-8") as file:
                        content = file.read()
                        lines = content.splitlines()
                        first_line = lines[0]
                        print(f"Sample line from file: {first_line}")
                        
                        for line in lines:
                            if line.strip():
                                index, text = parse_line(line)
                                if index and text:
                                    all_data.append((filename, index, text))
                                    
                        line_count = len(lines)
                        print(f"Total lines processed from {filename}: {line_count}")
            return all_data
        except Exception as e:
            print(f"Read data from text failed: {e}")
            return []

def main():
    print("Initializing Llama model and ChromaDB client...")
    embedding_function = CustomLlamaEmbeddingFunction()
    client = chromadb.PersistentClient(path="vectorDB/law_original_llama2")
    collection = client.get_or_create_collection(
        name="law_collection_llama_em2",
        embedding_function=embedding_function
    )
    
    directory_path = "data_full"
    # if we are embedding the summary content
    summary = True
    all_data = getDataFromText(directory_path, summary)
    print(f"Total entries to process: {len(all_data)}")


    # Batch process the documents with a smaller batch size due to model memory requirements
    batch_size = 50  # Reduced batch size for Llama model
    total_batches = (len(all_data) + batch_size - 1) // batch_size
    print(f"Total number of batches: {total_batches}")

    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches}")
        
        if summary:
            ids = [f"summary_{idx}" for idx, (law_name, _) in enumerate(batch, start=i)]
            documents = [content for _, content in batch]
            metadatas = [{"law_name": law_name} for law_name, _ in batch]
        else:
            ids = [f"{file}_{idx}" for file, idx, _ in batch]
            documents = [text for _, _, text in batch]
            metadatas = [{"file_name": file, "original_index": idx} for file, idx, _ in batch]

        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    print("Data successfully added to collection")
    print(f"Total documents added: {len(all_data)}")

if __name__ == "__main__":
    main()
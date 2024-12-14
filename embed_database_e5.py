import chromadb
from sentence_transformers import SentenceTransformer
import os
from chromadb.utils.embedding_functions import EmbeddingFunction

# Create a custom embedding function that wraps the sentence transformer
class CustomSentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
        self.instruction = """
        將法律條文轉換為可以匹配日常問題的表示方式。
        需要考慮：
        1. 將專業法律用語對應到一般民眾的日常用語
        2. 識別法條中描述的情境和實際案例的關聯
        3. 保留法條的核心概念和關鍵要素
        4. 連結相似的法律概念和情境
        請轉換以下法條：
        """
    def __call__(self, input: list[str]) -> list[list[float]]:
        # Format each text with our comprehensive instruction
        formatted_inputs = [f'Instruct: {self.instruction}\nInput: {text}' for text in input]
        
        embeddings = self.model.encode(formatted_inputs,
                                     convert_to_tensor=True,
                                     normalize_embeddings=True)
        
        return embeddings.tolist()

def parse_line(line: str) -> tuple:
    """Parse a single line into index and text."""
    parts = line.strip().split(' ', 1)
    print(parts)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None

def getDataFromText(directory_path: str, summary:bool = False) -> list:
    """Read data from multiple text files in the directory."""

    if summary:
        result = []

        # Read the file and process each line
        with open(directory_path, 'r', encoding='utf-8') as file:
            # Get the file name without the .txt extension
            law_name = os.path.basename(directory_path).replace('.txt', '')
            
            for line in file:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Add to result list [law_name, original_content]
                result.append((law_name, line.strip()))
        
        return result
    
    else:

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
    print("Initializing model and client...")
    embedding_function = CustomSentenceTransformerEmbeddingFunction()
    client = chromadb.PersistentClient(path="vectorDB/summary")
    collection = client.get_or_create_collection(
        name="law_collection",
        embedding_function=embedding_function
    )
    
    directory_path = "lawData/json_DFS" 

    summary = False
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
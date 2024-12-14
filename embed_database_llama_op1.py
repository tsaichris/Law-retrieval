import chromadb
from transformers import AutoTokenizer, AutoModel
import torch
import os
from chromadb.utils.embedding_functions import EmbeddingFunction
from huggingface_hub import login

# Authenticate with Hugging Face
login(token="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")

class CustomLlamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model_id = "lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.eval()  # Set model to evaluation mode
        

        # Similar instruction concept from the original code but adapted for the context
        self.instruction = ''
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        # Format inputs with instruction
        formatted_inputs = [f'{self.instruction}\n{text}' for text in input]
        
        all_embeddings = []
        # Process each text separately to manage memory
        for text in formatted_inputs:
            with torch.no_grad():
                # Tokenize and get model outputs
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                outputs = self.model(**inputs)
                
                # Get sentence embedding through mean pooling
                embeddings = outputs.last_hidden_state
                sentence_embedding = torch.mean(embeddings, dim=1).squeeze()
                
                # Convert to list and normalize
                embedding_list = sentence_embedding.tolist()
                # Normalize the embedding
                norm = sum(x*x for x in embedding_list) ** 0.5
                normalized_embedding = [x/norm for x in embedding_list]
                
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
        with open(directory_path, 'r', encoding='utf-8') as file:
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print("Initializing Llama model and ChromaDB client...")
    embedding_function = CustomLlamaEmbeddingFunction()
    client = chromadb.PersistentClient(path="vectorDB/law_original_llama1")
    collection = client.get_or_create_collection(
        name="law_collection_llama_em1",
        embedding_function=embedding_function
    )
    
    directory_path = "lawData/law_original"
    # if we are embedding the summary content
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
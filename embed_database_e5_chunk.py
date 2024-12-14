import chromadb
from sentence_transformers import SentenceTransformer
import os
from chromadb.utils.embedding_functions import EmbeddingFunction
from typing import List, Tuple
import textwrap
import logging
import time
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_chunks_with_overlap(text: str, content_max_length: int = 385, overlap: int = 50) -> List[str]:
    """Create overlapping chunks from text, accounting for instruction length."""
    logger.debug(f"Starting chunk creation with max_length={content_max_length}, overlap={overlap}")
    start_time = time.time()
    
    chunks = []
    start = 0
    text_length = len(text)
    logger.debug(f"Input text length: {text_length}")
    
    while start < text_length:
        # Calculate end position
        end = start + content_max_length
        
        # If this is not the last chunk, try to find a sentence boundary
        if end < text_length:
            # Look for the last sentence boundary within the chunk
            last_period = text.rfind('。', start, end)
            if last_period != -1:
                end = last_period + 1
                logger.debug(f"Found sentence boundary at position {last_period}")
            else:
                logger.debug(f"No sentence boundary found, using max length cut at {end}")
        else:
            end = text_length
            logger.debug("Processing final chunk")
        
        # Add the chunk
        chunk = text[start:end]
        chunks.append(chunk)
        logger.debug(f"Created chunk {len(chunks)} with length {len(chunk)}")
        
        # Calculate next start position with overlap
        start = end - overlap if end < text_length else text_length
    
    duration = time.time() - start_time
    logger.debug(f"Chunk creation completed in {duration:.2f} seconds, created {len(chunks)} chunks")
    return chunks

def parse_line(line: str) -> tuple:
    """Parse a single line into index and text."""
    parts = line.strip().split(' ', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    logger.warning(f"Skipping malformed line: {line}")
    return None, None

def getDataFromText(directory_path: str, summary:bool = False) -> list:
    """Read data from multiple text files in the directory."""
    logger.info(f"Starting to read data from: {directory_path}")
    
    if summary:
        result = []
        try:
            with open(directory_path, 'r', encoding='utf-8') as file:
                law_name = os.path.basename(directory_path).replace('.txt', '')
                logger.info(f"Processing summary file: {law_name}")
                
                for line in file:
                    if line.strip():
                        result.append((law_name, line.strip()))
                
                logger.info(f"Processed {len(result)} lines from summary file")
                return result
        except Exception as e:
            logger.error(f"Error processing summary file: {e}")
            return []
    else:
        all_data = []
        try:
            logger.info(f"Reading files from directory: {directory_path}")
            files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
            logger.info(f"Found {len(files)} text files")
            
            for filename in files:
                file_path = os.path.join(directory_path, filename)
                logger.info(f"Processing file: {filename}")
                
                try:
                    with open(file_path, 'r', encoding="utf-8") as file:
                        content = file.read()
                        lines = content.splitlines()
                        processed_lines = 0
                        
                        for line in lines:
                            if line.strip():
                                index, text = parse_line(line)
                                if index and text:
                                    all_data.append((filename, index, text))
                                    processed_lines += 1
                        
                        logger.info(f"Processed {processed_lines} valid lines from {filename}")
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {e}")
                    continue
            
            logger.info(f"Total processed entries: {len(all_data)}")
            return all_data
        except Exception as e:
            logger.error(f"Read data from text failed: {e}")
            return []
class CustomSentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_max_length: int = 512):
        logger.info("Initializing CustomSentenceTransformerEmbeddingFunction")
        try:
            self.model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
        self.model_max_length = model_max_length
        self.instruction = """
        將法律條文轉換為可以匹配日常問題的表示方式。
        需要考慮：
        1. 將專業法律用語對應到一般民眅的日常用語
        2. 識別法條中描述的情境和實際案例的關聯
        3. 保留法條的核心概念和關鍵要素
        4. 連結相似的法律概念和情境
        請轉換以下法條：
        """
        self.instruction_length = 117
        self.max_content_length = self.model_max_length - self.instruction_length - 10
        logger.info(f"Max content length set to: {self.max_content_length}")

    def __call__(self, input: List[str]) -> List[List[float]]:
        start_time = time.time()
        all_embeddings = []
        logger.info(f"Processing {len(input)} inputs for embedding")
        
        try:
            for idx, text in enumerate(input):
                logger.info(f"Processing input {idx}, length: {len(text)}")
                
                # Create chunks
                chunks = create_chunks_with_overlap(text, self.max_content_length)
                logger.info(f"Input {idx}: Split into {len(chunks)} chunks")
                
                # Format chunks with instruction
                formatted_chunks = [f'Instruct: {self.instruction}\nInput: {chunk}' for chunk in chunks]
                logger.info(f"Created {len(formatted_chunks)} formatted chunks")
                
                try:
                    # Generate embeddings
                    chunk_start_time = time.time()
                    chunk_embeddings = self.model.encode(formatted_chunks,
                                                       convert_to_tensor=True,
                                                       normalize_embeddings=True)
                    chunk_duration = time.time() - chunk_start_time
                    logger.info(f"Generated embeddings for input {idx} in {chunk_duration:.2f} seconds")
                    
                    # Combine embeddings if necessary
                    if len(chunks) > 1:
                        final_embedding = chunk_embeddings.mean(dim=0)
                        logger.info(f"Averaged {len(chunks)} chunk embeddings")
                    else:
                        final_embedding = chunk_embeddings[0]
                    
                    all_embeddings.append(final_embedding.tolist())
                    
                except Exception as e:
                    logger.error(f"Error embedding chunks for input {idx}: {e}")
                    raise
                
        except Exception as e:
            logger.error(f"Error in embedding generation: {e}")
            raise
            
        duration = time.time() - start_time
        logger.info(f"Embedding generation completed in {duration:.2f} seconds")
        return all_embeddings


class LawDocumentManager:
    def __init__(self, db_path: str = "vectorDB/law_chunks"):
        logger.info(f"Initializing LawDocumentManager with path: {db_path}")
        try:
            self.embedding_function = CustomSentenceTransformerEmbeddingFunction()
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(
                name="law_collection",
                embedding_function=self.embedding_function
            )
            self.max_content_length = self.embedding_function.max_content_length
            logger.info("LawDocumentManager initialization complete")
        except Exception as e:
            logger.error(f"Error initializing LawDocumentManager: {e}")
            raise

    def add_documents(self, documents: List[Tuple[str, str, str]], overlap: int = 50):
        """Add documents to the collection with chunking."""
        start_time = time.time()
        logger.info(f"Starting to add {len(documents)} documents")
        
        total_chunks = 0
        for doc_id, (filename, index, text) in enumerate(tqdm(documents, desc="Processing documents")):
            chunk_start_time = time.time()
            try:
                logger.debug(f"Processing document {doc_id}: {filename}_{index}")
                logger.debug(f"Document length: {len(text)}")
                
                # Create chunks
                chunks = create_chunks_with_overlap(text, 
                                                  content_max_length=self.max_content_length, 
                                                  overlap=overlap)
                
                logger.debug(f"Document {filename}_{index} split into {len(chunks)} chunks")
                
                # Add chunks to collection
                for chunk_id, chunk in enumerate(chunks):
                    chunk_add_start = time.time()
                    try:
                        logger.debug(f"Adding chunk {chunk_id}/{len(chunks)} for document {filename}_{index}")
                        self.collection.add(
                            documents=[chunk],
                            metadatas=[{
                                "file_name": filename,
                                "original_index": index,
                                "chunk_id": chunk_id,
                                "total_chunks": len(chunks)
                            }],
                            ids=[f"{filename}_{index}_chunk_{chunk_id}"]
                        )
                        chunk_add_duration = time.time() - chunk_add_start
                        logger.debug(f"Added chunk {chunk_id} in {chunk_add_duration:.2f} seconds")
                        total_chunks += 1
                        
                    except Exception as e:
                        logger.error(f"Error adding chunk {chunk_id} for document {filename}_{index}: {e}")
                        logger.error(f"Chunk content length: {len(chunk)}")
                        logger.error(f"Chunk content: {chunk[:100]}...")
                        continue
                
                chunk_duration = time.time() - chunk_start_time
                logger.debug(f"Processed document {doc_id} in {chunk_duration:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error processing document {filename}_{index}: {e}")
                continue
        
        duration = time.time() - start_time
        logger.info(f"Added {total_chunks} chunks from {len(documents)} documents in {duration:.2f} seconds")

def main():
    try:
        logger.info("Starting main processing")
        db_path = "vectorDB/law_original_chunk"
        doc_manager = LawDocumentManager(db_path=db_path)
        
        directory_path = "lawData/law_original"
        logger.info("Reading data from text files")
        all_data = getDataFromText(directory_path)
        logger.info(f"Total documents to process: {len(all_data)}")
        
        batch_size = 1000
        total_batches = (len(all_data) + batch_size - 1) // batch_size
        logger.info(f"Processing {total_batches} batches of size {batch_size}")
        
        for i in range(0, len(all_data), batch_size):
            batch = all_data[i:i + batch_size]
            current_batch = i // batch_size + 1
            logger.info(f"Processing batch {current_batch}/{total_batches}")
            
            start_time = time.time()
            doc_manager.add_documents(batch)
            duration = time.time() - start_time
            logger.info(f"Batch {current_batch} completed in {duration:.2f} seconds")
            
            # Add a small delay between batches to prevent overload
            time.sleep(1)
        
        logger.info("Processing completed successfully")
    
    except Exception as e:
        logger.error(f"Main processing failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import re
def get_detailed_instruct(task_description: str, query: str) -> str:
    """
    Formats the task description and query into a standardized instruction format.
    
    Args:
        task_description: A string describing the task or context
        query: The user's search query
        
    Returns:
        A formatted string combining the task description and query
    """
    return f'Instruct: {task_description}\nQuery: {query}'

def reform_results(results_list, logger):
    reformed_list = []
    
    for result in results_list:
        # Extract rank value
        rank = result['rank']
        # Extract index value
        index_value = result['index']
        # Extract article numbers
        test_part = result['text']
        parts = test_part.split(',')
        article_numbers = []
        for part in parts:
            matches = re.search(r'(\d+(?:-\d+)?)\s*條', part.strip())
            if matches:
                number = matches.group(1)
                # Validate number format
                try:
                    number_parts = number.split('-')
                    valid = all(0 <= int(num) <= 9999 for num in number_parts)
                    if valid:
                        article_numbers.append(number)
                except ValueError:
                    continue

        # Only add to reformed list if we found valid article numbers
        if rank is not None and index_value is not None and article_numbers:
            reformed_dict = {
                "rank": rank,
                "law_name": index_value,
                "index": article_numbers
            }
            reformed_list.append(reformed_dict)
            #logger.info(f"\nRank {reformed_dict['rank']}:")
            #logger.info(f"law_name: {reformed_dict['law_name']}")
            #logger.info(f"index: {reformed_dict['index']}")
    
    return reformed_list

def get_by_file_and_indices(law_name: str, index: list, formatted_results:list, logger:object, collection:object) -> list:
    """
    Retrieve documents by file name and specific index using metadata filtering.
    example:
    index = ["1", "2", "3-1"]
    """
    try:
        logger.info(f"Retrieving documents from file: {law_name} with index: {index}")
        
        
        # Get all documents for the file first
        results = collection.get(
            where={"file_name": f"{law_name}.txt"},  # Add .txt extension based on the test results
            include=['documents', 'metadatas']
        )
        
        # Filter the results for matching indices
        if results and results['documents']:
            for i in range(len(results['documents'])):
                if results['metadatas'][i]['original_index'] in index:
                    result = {
                        "file": results['metadatas'][i]['file_name'],
                        "index": results['metadatas'][i]['original_index'],
                        "text": results['documents'][i]
                    }
                    formatted_results.append(result)
                    
                    #logger.info(f"\nlaw_name {result['file']}:")
                    #logger.info(f"index: {result['index']}")
                    #logger.info(f"index: {result['text']}")
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        return []    
def search_chunk(search_result, n_results):
    # Aggregate chunks by document
    doc_results = {}
    for i in range(len(search_result['documents'][0])):
        metadata = search_result['metadatas'][0][i]
        doc_key = f"{metadata['file_name']}_{metadata['original_index']}"
        
        if doc_key not in doc_results:
            doc_results[doc_key] = {
                'file': metadata['file_name'],
                'index': metadata['original_index'],
                'chunks': [],
                'distances': [],
                'embeddings': search_result['embeddings'][0][i]
            }
        
        doc_results[doc_key]['chunks'].append(search_result['documents'][0][i])
        doc_results[doc_key]['distances'].append(search_result['distances'][0][i])

    # Format final results
    final_results = []
    rank = 1
    
    # Sort by best (minimum) distance across chunks
    sorted_docs = sorted(
        doc_results.values(),
        key=lambda x: min(x['distances'])
    )[:n_results]
    
    for doc in sorted_docs:
        # Combine chunks if there are multiple
        combined_text = ' '.join(doc['chunks'])
        
        result = {
            "rank": rank,
            "file": doc['file'],
            "index": doc['index'].rstrip(","),
            "text": combined_text,
            "distance": min(doc['distances'])  # Use best matching chunk's distance
        }
        final_results.append(result)
        rank += 1
    
    return final_results
class LawSearcher_e5:
    """
    A class for searching legal documents using the E5 multilingual embedding model.
    
    This class provides functionality to:
    - Initialize embedding model and ChromaDB client
    - Search for relevant legal documents based on query
    - Process and rank search results
    """
    
    def __init__(self, client_path:str = "test_db_full", use_summary:bool = False, client_path_summary: str = None):
        """
        Initialize the LawSearcher with specified client path and setup logging.
        
        Args:
            client_path: Path to the ChromaDB client database
        """
        self.use_summary = use_summary
        self.client_path_summary = client_path_summary
        self.client_summary = chromadb.PersistentClient(path=client_path_summary)
        self.collection_summary = self.client_summary.get_collection(name="law_collection")

        # Setup logging configuration
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler('logs/searcher_result.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        #self.logger.addHandler(logging.StreamHandler()) # print in terminal
        self.logger.setLevel(logging.INFO)

        self.logger.info("Initializing model and client...")
        self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
        self.client = chromadb.PersistentClient(path=client_path)
        self.collection = self.client.get_collection(name="law_collection")



    def search(self, query: str, n_results_original: int = 25, n_results_summary: int = 25, chunk: bool = False):
        """
        Search for relevant legal documents based on the input query.
        
        Args:
            query: The search query string
            n_results: Number of results to return (default: 5)
            
        Returns:
            List of dictionaries containing ranked search results with metadata
        """
        try:
            self.logger.info("Processing search query...")
            self.logger.info(f"Query: {query}")
            
            #query_task = "Given a situation or question, find relevant laws and regulations"
            query_task = '''
            任務：搜尋與此情況相關的法律條文和規範。
            背景：搜索法律文件以找出與查詢最相關的法律、法規和條文。同時考慮法律術語的
            精確匹配和法律概念的語意相關性。需特別注意條文編號和法規名稱的對應關係。
            '''
            """
            Enhanced instruction formatting for legal document retrieval
            
            Version 1 - English, Detailed:
            task_description = '''
            Task: Find relevant legal articles and regulations that address this situation.
            Context: Search through legal documents to find specific laws, regulations, and articles 
            that are most relevant to the query. Consider both exact terminology matches and 
            semantic relevance to legal concepts.
            '''

            Version 2 - Chinese, Detailed:
            task_description = '''
            任務：搜尋與此情況相關的法律條文和規範。
            背景：搜索法律文件以找出與查詢最相關的法律、法規和條文。同時考慮法律術語的
            精確匹配和法律概念的語意相關性。需特別注意條文編號和法規名稱的對應關係。
            '''

            Version 3 - Bilingual, Concise:
            task_description = '''
            Task/任務: Match the query to relevant legal articles and regulations
            搜尋對應的法律條文並建立與查詢內容的關聯性
            '''
            """
            query_with_instruct = get_detailed_instruct(query_task, query)
            
            self.logger.info("Formatted query with instruction:")
            self.logger.info(query_with_instruct)
            
            query_vector = self.embedding_model.encode(query_with_instruct, 
                                                     convert_to_tensor=True, 
                                                     normalize_embeddings=True)
            
            self.logger.info(f"Query embedding shape: {query_vector.shape}")
            self.logger.debug(f"Sample of query embedding (first 5 values): {query_vector[:5]}")
            
            self.logger.info("Query Results:")
            self.logger.info("--------------")

            


            if self.use_summary:
                res_summary = self.collection_summary.query(
                    query_embeddings=[query_vector.tolist()],
                    n_results=n_results_summary,
                    include=['distances', 'embeddings', 'documents', 'metadatas']
                )
                if chunk:
                    results_summary = search_chunk(res_summary, n_results_summary)
                else:
                    results_summary = []
                    for i in range(len(res_summary['documents'][0])):
                        result = {
                            "rank": i + 1,
                            "file": res_summary['metadatas'][0][i]['file_name'],
                            "index": res_summary['metadatas'][0][i]['original_index'].rstrip(","),
                            "text": res_summary['documents'][0][i],
                            "distance": res_summary['distances'][0][i]
                        }
                        results_summary.append(result)
            
            res_ori = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=n_results_original,
                include=['distances', 'embeddings', 'documents', 'metadatas']
            )
            if chunk:
                results_original = search_chunk(res_ori, n_results_original)
            else:
                results_original = []
                for i in range(len(res_ori['documents'][0])):
                    result = {
                        "rank": i + 1,
                        "file": res_ori['metadatas'][0][i]['file_name'],
                        "index": res_ori['metadatas'][0][i]['original_index'].rstrip(","),
                        "text": res_ori['documents'][0][i],
                        "distance": res_ori['distances'][0][i]
                    }
                    results_original.append(result)
                    if not self.use_summary:
                        self.logger.info(f"\nRank {i+1}:")
                        self.logger.info(f"File: {result['file']}")
                        self.logger.info(f"Index: {result['index']}")
                        self.logger.info(f"Text: {result['text']}")
                        self.logger.info(f"Distance: {result['distance']}")         

            if self.use_summary:
                reformed_list = reform_results(results_summary, self.logger)
                summary_results = []
                for dic in reformed_list:
                    law_name = dic['law_name']
                    index =dic['index']
                    
                    summary_results = get_by_file_and_indices(law_name, index,summary_results, self.logger, self.collection)
                return summary_results, results_original

        
            else:
                return results_original
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []

def main():
    """
    Main function to demonstrate the usage of LawSearcher_e5.
    Sets up the searcher and performs a sample search query.
    """
    searcher = LawSearcher_e5(client_path = 'vectorDB/law_original',use_summary = True, client_path_summary='vectorDB/summary')

    query = '請問依農發條例由農會輔導設立的農業產銷班是法人或機關團體嗎？有一原住民農業產銷班有訴訟需求欲申請法律扶助，但依法律扶助法及法律扶助施行範圍辦法地2條規定，法人或機關團體，本會不予扶助。請問農業產銷班的法律地位為何？是法人或機關團體嗎？'

    searcher.search(query)

if __name__ == "__main__":
    main()
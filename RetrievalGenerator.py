from utli.LawSearcher_llama import LawSearcher_llama
from utli.LawSearcher_e5 import LawSearcher_e5
import logging
from typing import List, Dict, Optional
from utli.format_law import format_law_reference
from utli.reranker import use_reranker
from tqdm import tqdm
import json

def safe_str_strip(value: Optional[str]) -> str:
    """Safely strip a string that might be None."""
    if value is None:
        return ""
    return str(value).strip()

def union_retrieved_docs(summary_doc, original_retrieved_docs):
    """
    Merges two lists of document dictionaries based on matching 'file' and 'index' values.
    Excludes 'rank' and 'distance' fields from the final result.
    
    Args:
        summary_doc: List of dicts with 'file', 'index', and 'text' keys
        original_retrieved_docs: List of dicts with 'file', 'index', 'text', 'rank', and 'distance' keys
        
    Returns:
        List of merged unique documents containing only 'file', 'index', and 'text' fields
    """
    seen = set()
    result = []
    
    # Process all documents from both lists
    for doc in summary_doc + original_retrieved_docs:
        # Create unique key for comparison
        key = (doc['file'], doc['index'])
        
        if key not in seen:
            seen.add(key)
            # Only keep relevant fields
            result.append({
                'file': doc['file'],
                'index': doc['index'],
                'text': doc['text']
            })
    
    return result

class InitialRetrievalGenerator:
    """
    Generates initial retrievals for both training and testing scenarios.
    Handles statistics calculation for training data when labels are available.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler('logs/Initial_Retrieval_generation.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        #self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)
    
    def generate_retrievals(self, 
                          searcher: object, 
                          processed_data: List[Dict], 
                          is_training: bool = True,
                          n_candidates_original: int = 25,
                          n_candidates_summary: int = 25,
                          n_final: int = 5, 
                          useReranker: bool = False,
                          use_summary: bool = False, 
                          reranker_path: str = None) -> Dict:
        """
        Generate retrievals for queries, with optional accuracy calculation for training data.
        
        Args:
            searcher: Search object instance
            processed_data: List of processed examples
            is_training: Whether processing training data (with labels) or test data
            n_candidates: Number of candidates to retrieve per query
            n_final: Final number of results after reranking
            useReranker: Whether to use reranker
            reranker_path: Path to reranker model
            
        Returns:
            Dictionary mapping query IDs to retrieved documents
        """
        self.logger.info(f"Starting retrieval generation for {len(processed_data)} queries")
        initial_retrievals = {}
        
        # Statistics tracking (for all data, but only used in training mode)
        total_queries = len(processed_data)
        queries_with_hits = 0
        total_coverage_rate = 0
        #for entry in tqdm(processed_data, desc="Processing queries", position=0, leave=True):
        for entry in processed_data:   

            try:
                # Get candidates using the search method
                #print('retrievaling')
                if use_summary:
                    summary_doc, original_retrieved_docs  = searcher.search(query=entry['query_text'], n_results_original = n_candidates_original, n_results_summary = n_candidates_summary)
                    retrieved_docs = union_retrieved_docs(summary_doc, original_retrieved_docs)
                else:
                    retrieved_docs,  = searcher.search(query=entry['query_text'], n_results_original = n_candidates_original)
                #print(retrieved_docs)
                if useReranker:
                    retrieved_docs = use_reranker(
                        entry['query_text'], 
                        reranker_path, 
                        initial_search_results=retrieved_docs,
                        n_final=n_final
                    )
                #print('checking')
                if retrieved_docs:
                    initial_retrievals[entry['id']] = retrieved_docs
                    
                    # Calculate statistics only for training data
                    if is_training and 'labels' in entry:
                        # Process statistics directly here as in original code
                        correct_laws = set(entry['labels'].split(',')) if isinstance(entry['labels'], str) else set(entry['labels'])
                        hits = 0
                        
                        self.logger.debug(f"\nQuery ID {entry['id']}:")
                        self.logger.debug(f"Expected laws: {correct_laws}")

                        for doc in retrieved_docs:
                            formatted_ref = format_law_reference(doc['file'], doc['index'])
                            #print(formatted_ref)
                            if formatted_ref in correct_laws:
                                hits += 1
                                self.logger.debug(f"Found match: {formatted_ref}")
                        
                        query_coverage_rate = (hits / len(correct_laws)) * 100 if len(correct_laws) > 0 else 0
                        total_coverage_rate += query_coverage_rate
                        
                        if hits > 0:
                            queries_with_hits += 1
                        
                        self.logger.info(
                            f"Query {entry['id']}: {hits}/{len(correct_laws)} laws found, "
                            f"coverage rate: {query_coverage_rate:.2f}%"
                        )
                        
                        if hits == 0:
                            self.logger.debug("Sample of attempted matches:")
                            for i, doc in enumerate(retrieved_docs[:3]):
                                formatted_ref = format_law_reference(doc['file'], doc['index'])
                                self.logger.debug(f"Attempted match: {formatted_ref}")
                    else:
                        self.logger.debug(f"Processed test query {entry['id']}")
                        
            except Exception as e:
                self.logger.error(f"Error processing query {entry['id']}: {e}")
                continue
        
        # Log statistics only for training data
        if is_training:
            self._log_retrieval_statistics(total_queries, queries_with_hits, total_coverage_rate)
        
        return initial_retrievals
    
    def _log_retrieval_statistics(self, 
                                total_queries: int, 
                                queries_with_hits: int,
                                total_coverage_rate: float):
        """Log overall retrieval statistics"""
        statistics = {
            "total_queries": total_queries,
            "queries_with_hits": queries_with_hits,
            "average_coverage_rate": total_coverage_rate/total_queries if total_queries > 0 else 0
        }
        
        self.logger.info("\nRetrieval Statistics Summary:")
        for key, value in statistics.items():
            if key == "average_coverage_rate":
                self.logger.info(f"{key}: {value:.2f}%")
            else:
                self.logger.info(f"{key}: {value}")
                
        try:
            with open('logs/retrieval_statistics.txt', 'a') as stats_file:
                for key, value in statistics.items():
                    if key == "average_coverage_rate":
                        stats_file.write(f"{key}: {value:.2f}%\n")
                    else:
                        stats_file.write(f"{key}: {value}\n")
        except Exception as e:
            self.logger.error(f"Error writing to statistics file: {e}")

def process_json_line(line: str) -> Optional[Dict]:
    """Process a single JSON line and return a processed data entry or None if invalid."""
    try:
        data = json.loads(line)
        
        # Get title and question, ensuring they're strings
        title = safe_str_strip(data.get('title'))
        question = safe_str_strip(data.get('question'))
        
        # Construct query text
        if title and question:
            data['query_text'] = f"{title} [SEP] {question}"
        elif title:
            data['query_text'] = title
        elif question:
            data['query_text'] = question
        else:
            print(f"Warning: Entry {data.get('id', 'unknown')} has no title or question")
            return None
            
        return data
        
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON line: {line[:100]}...")
        return None
    except Exception as e:
        print(f"Warning: Error processing line: {str(e)}")
        return None

def stage1RE_generator():
    """
    Generate retrievals for either training or test data.
    
    Args:
        is_training: Whether to process training data (with labels) or test data
    """
    generator = InitialRetrievalGenerator()
    reranker_path = 'models_reranker/best_model'
    client_path = 'vectorDB/law_original'
    client_path_summary='vectorDB/summary'
    use_summary = True
    searcher = LawSearcher_e5(client_path=client_path, use_summary=use_summary, client_path_summary=client_path_summary)
    #searcher = LawSearcher_llama(client_path)
    n_initial_summary = 15
    n_initial_original = 25
    n_final = 25
    useReranker = False
    is_training=True

    # Load appropriate data file
    data_file = 'trainingData/processed_training_data.json' if is_training else 'trainingData/test_data.json'
    processed_data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        if is_training:
            processed_data = json.load(f)
        else:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                processed_line = process_json_line(line)
                if processed_line is not None:
                    processed_data.append(processed_line)
                else:
                    print(f"Skipping invalid line {line_number}")

    if not processed_data:
        raise ValueError(f"No valid data could be loaded from {data_file}")
        
    print(f"Successfully loaded {len(processed_data)} entries from {data_file}")
    
    # Generate retrievals
    initial_retrievals = generator.generate_retrievals(
        searcher=searcher,
        processed_data=processed_data,
        is_training=is_training,
        n_candidates_original =n_initial_original,
        n_candidates_summary =n_initial_summary,
        n_final=n_final,
        useReranker=useReranker,
        use_summary = use_summary,
        reranker_path=reranker_path
    )
    
    if not initial_retrievals:
        raise ValueError("No retrievals were generated")
    
    # Save results
    output_file = 'retrival_result.json' if is_training else 'test_retrival_result.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(initial_retrievals, f, ensure_ascii=False, indent=2)
    print(f"Saved retrievals to {output_file}")

if __name__ == "__main__":
    # By default process training data, set is_training=False for test data
    stage1RE_generator()
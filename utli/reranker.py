import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict
from utli.LawSearcher_e5 import LawSearcher_e5

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:  # Only add handlers if they don't exist
    handler = logging.FileHandler('logs/reranker_result.log')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

class Reranker_bert:
    """
    A wrapper class for using a trained reranker model for inference.
    This class loads a previously trained model and provides methods for reranking documents.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the reranker with a trained model.
        
        Args:
            model_path: Base path to the saved model and tokenizer
                       (will append '_model.pt' and '_tokenizer' automatically)
        """
        self.max_length = 512
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing trained reranker on device: {self.device}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(f'{model_path}_tokenizer')
            
            # Initialize model architecture
            self.model = AutoModelForSequenceClassification.from_pretrained(
                'bert-base-chinese',
                num_labels=1
            ).to(self.device)
            
            # Load trained weights
            checkpoint = torch.load(f'{model_path}_model.pt', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print("Successfully loaded trained model and tokenizer")
            
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise

    def rerank(self, query: str, initial_results: List[Dict], 
               batch_size: int = 16) -> List[Dict]:
        """
        Rerank the initial retrieval results using the trained model.
        
        Args:
            query: The original query string
            initial_results: List of documents from initial retrieval
            batch_size: Batch size for processing documents
            
        Returns:
            List of reranked documents with scores
        """
        try:
            reranking_scores = []
            
            # Process documents in batches for efficiency
            for i in range(0, len(initial_results), batch_size):
                batch_docs = initial_results[i:i + batch_size]
                batch_scores = self._score_batch(query, batch_docs)
                reranking_scores.extend(zip(batch_scores, batch_docs))
            
            # Sort by scores in descending order
            reranked_results = [
                {**doc, 'rerank_score': score} 
                for score, doc in sorted(reranking_scores, reverse=True)
            ]
            
            return reranked_results
            
        except Exception as e:
            print(f"Error during reranking: {e}")
            return initial_results  # Fall back to initial results if reranking fails
    
    def _score_batch(self, query: str, docs: List[Dict]) -> List[float]:
        """Score a batch of documents against the query"""
        try:
            with torch.no_grad():
                # Prepare inputs
                inputs = [
                    self.tokenizer(
                        query,
                        doc['text'],
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=512
                    ) for doc in docs
                ]
                
                # Combine into batch
                batch_inputs = {
                    'input_ids': torch.cat([inp['input_ids'] for inp in inputs]),
                    'attention_mask': torch.cat([inp['attention_mask'] for inp in inputs])
                }
                
                # Move to device
                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                
                # Get model predictions
                outputs = self.model(**batch_inputs)
                scores = torch.sigmoid(outputs.logits).squeeze().tolist()
                
                # Handle single document case
                if not isinstance(scores, list):
                    scores = [scores]
                
                return scores
                
        except Exception as e:
            print(f"Error scoring batch: {e}")
            return [0.0] * len(docs)  # Return neutral scores on error
        


def use_reranker(query:str, reranker_path:str, initial_search_results:list, n_final:int):
    
    def log_search_results(results: List[Dict]):
        """Log search results in a formatted way"""
        logger.info("\nSearch Results:")
        logger.info("--------------")
        for i, result in enumerate(results, 1):
            logger.info(f"\nRank {i}:")
            logger.info(f"File: {result['file']}")
            logger.info(f"Index: {result['index']}")
            logger.info(f"Text: {result['text']}")
            if 'rerank_score' in result:
                logger.info(f"Reranking Score: {result['rerank_score']:.4f}")

    reranker = Reranker_bert(model_path = reranker_path)
    try:
        reranked_results = reranker.rerank(query, initial_search_results)

        # Get top results
        final_results = reranked_results[:n_final]
        
        # Log results
        log_search_results(final_results)
        
        return final_results
        
    except Exception as e:
        logger.error(f"Error in enhanced search: {e}")
        return []



if __name__ == "__main__":
    try:
        # Initialize enhanced searcher
        reranker_path='models_reranker/best_model'
        client_path = 'test_db_full'
        searcher = LawSearcher_e5(client_path = client_path)
        n_final = 5
        # Example query
        query = '帶手指虎打傷人，手指虎會被「沒收」還是「沒入」'
        initial_results = searcher.search(query, n_results = 50)
        
        if initial_results:
            print("Successfully retrieved initaial results")
        else:
            print("No results found for query")  

        
        reranker = use_reranker(query, reranker_path, initial_results,n_final)
        print("Successfully re-ranked initaial results")

            
    except Exception as e:
        print(f"Error in inference: {e}")
        raise

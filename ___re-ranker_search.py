import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict
from utli.LawSearcher_e5 import LawSearcher  # Import your existing search code

class TrainedLegalReranker:
    def __init__(self, model_path: str):
        """Initialize the reranker with a trained model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path + '_tokenizer')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-chinese',
            num_labels=1
        ).to(self.device)
        
        # Load trained weights
        checkpoint = torch.load(model_path + '_model.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def rerank(self, query: str, initial_results: List[Dict]) -> List[Dict]:
        """Rerank the initial retrieval results"""
        reranking_scores = []
        
        with torch.no_grad():
            for doc in initial_results:
                # Prepare input
                inputs = self.tokenizer(
                    query,
                    doc['text'],
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get prediction score
                score = self.model(**inputs).logits.squeeze().item()
                reranking_scores.append((score, doc))
        
        # Sort by scores
        reranked_results = [doc for _, doc in sorted(reranking_scores, reverse=True)]
        return reranked_results

class EnhancedLawSearcher(LawSearcher):
    def __init__(self, reranker_path: str):
        """Initialize with both embedding search and trained reranker"""
        super().__init__()
        self.reranker = TrainedLegalReranker(reranker_path)
    
    def search(self, query: str, n_final_results: int = 5) -> List[Dict]:
        """Perform two-stage retrieval"""
        try:
            print("\nProcessing search query...")
            print(f"Query: {query}")
            
            # First stage: Get initial candidates
            initial_results = self._vector_search(query, n_results=20)
            
            # Second stage: Rerank results
            reranked_results = self.reranker.rerank(query, initial_results)
            
            # Get top results
            final_results = reranked_results[:n_final_results]
            
            # Print results
            self._print_results(final_results)
            
            return final_results
            
        except Exception as e:
            print(f"Search failed: {e}")
            return []
    
    def _print_results(self, results: List[Dict]):
        """Print search results in a formatted way"""
        print("\nSearch Results:", "\n--------------")
        for i, result in enumerate(results, 1):
            print(f"\nRank {i}:")
            print(f"File: {result['file']}")
            print(f"Index: {result['index']}")
            print(f"Text: {result['text']}")

def main():
    # Initialize enhanced searcher with trained reranker
    searcher = EnhancedLawSearcher(reranker_path='legal_reranker')
    
    # Example query
    query = '帶手指虎打傷人，手指虎會被「沒收」還是「沒入」'
    results = searcher.search(query)

if __name__ == "__main__":
    main()
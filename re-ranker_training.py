import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import logging
from datetime import datetime
from utli.format_law import format_law_reference
from RetrievalGenerator import InitialRetrievalGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/reranker_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



class TrainingDataCache:
    """Manages caching and loading of prepared training examples"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def save_examples(self, examples: List[Dict], metadata: Dict, 
                     filename: str = "training_examples"):
        """
        Save training examples and metadata to cache.
        
        Args:
            examples: List of prepared training examples
            metadata: Dictionary containing relevant metadata (e.g., max_length)
            filename: Base filename for the cache
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_path = os.path.join(self.cache_dir, f"{filename}_{timestamp}.pt")
        
        # Convert examples to a format suitable for saving
        saved_examples = []
        for ex in examples:
            saved_ex = {
                'input_ids': ex['inputs']['input_ids'].cpu(),
                'attention_mask': ex['inputs']['attention_mask'].cpu(),
                'label': ex['label'].cpu()
            }
            saved_examples.append(saved_ex)
        
        # Save both examples and metadata
        torch.save({
            'examples': saved_examples,
            'metadata': metadata
        }, cache_path)
        
        logger.info(f"Saved {len(examples)} training examples to {cache_path}")
        return cache_path
    
    def load_examples(self, cache_path: str) -> Tuple[List[Dict], Dict]:
        """
        Load training examples from cache.
        
        Returns:
            Tuple of (training_examples, metadata)
        """
        logger.info(f"Loading training examples from {cache_path}")
        
        cache = torch.load(cache_path)
        examples = cache['examples']
        metadata = cache['metadata']
        
        # Convert back to original format
        training_examples = []
        for ex in examples:
            training_ex = {
                'inputs': {
                    'input_ids': ex['input_ids'],
                    'attention_mask': ex['attention_mask']
                },
                'label': ex['label']
            }
            training_examples.append(training_ex)
        
        logger.info(f"Loaded {len(training_examples)} training examples")
        return training_examples, metadata




class LegalReranker:
    """
    Reranker model for legal document retrieval.
    Uses BERT-based model to rerank initial retrieval results.
    """
    
    def __init__(self, model_save_dir: str = "models_reranker"):
        """
        Initialize the reranker model.
        
        Args:
            model_save_dir: Directory to save model checkpoints
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.max_length = 512
        self.model_save_dir = model_save_dir
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-chinese',
            num_labels=1
        ).to(self.device)
        
        logger.info("Initialized LegalReranker with BERT model")



    def analyze_training_data(self, processed_data: List[Dict], 
                            initial_retrievals: Dict) -> Tuple[int, int, int]:
        """
        Analyze training data distribution and quality.
        
        Returns:
            Tuple of (total_examples, positive_examples, negative_examples)
        """
        logger.info("Analyzing training data distribution...")
        total_examples = 0
        positive_examples = 0
        examples_per_query = []
        
        for entry in processed_data:
            correct_laws = set(entry['labels'])
            retrieved_docs = initial_retrievals.get(entry['id'], [])
            
            for doc in retrieved_docs:
                total_examples += 1
                formatted_ref = format_law_reference(doc['file'], doc['index'])
                if formatted_ref in correct_laws:
                    positive_examples += 1
            
            if retrieved_docs:
                examples_per_query.append(len(retrieved_docs))
        
        negative_examples = total_examples - positive_examples
        self._log_training_data_analysis(total_examples, positive_examples, 
                                       negative_examples, examples_per_query)
        
        return total_examples, positive_examples, negative_examples
    
    def prepare_text_pair(self, query: str, doc_text: str) -> Tuple[List[int], List[int]]:
        """
        Prepare a query-document pair for BERT input with careful length handling.
        
        This method implements an intelligent truncation strategy that:
        1. Preserves as much of the query as possible (since it contains key search terms)
        2. Allocates remaining space to the document
        3. Accounts for special tokens ([CLS] and [SEP])
        
        Args:
            query: The search query text
            doc_text: The document text
            
        Returns:
            Tuple of (input_ids, attention_mask)
        """
        # Calculate space needed for special tokens:
        # [CLS] at start, [SEP] after query, [SEP] at end = 3 tokens
        special_tokens_count = 3
        max_total_length = 512 - special_tokens_count
        
        # First, tokenize both texts separately
        query_tokens = self.tokenizer.encode(query, add_special_tokens=False)
        doc_tokens = self.tokenizer.encode(doc_text, add_special_tokens=False)
        
        # Reserve space for query (up to 128 tokens, adjust if needed)
        max_query_length = 128
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[:max_query_length]
            logger.debug(f"Query truncated from {len(query_tokens)} to {max_query_length} tokens")
        
        # Calculate remaining space for document
        remaining_length = max_total_length - len(query_tokens)
        
        # Truncate document if necessary
        if len(doc_tokens) > remaining_length:
            doc_tokens = doc_tokens[:remaining_length]
            logger.debug(f"Document truncated to {remaining_length} tokens")
        
        # Construct the full sequence
        input_ids = (
            [self.tokenizer.cls_token_id]  # [CLS]
            + query_tokens  # Query
            + [self.tokenizer.sep_token_id]  # [SEP]
            + doc_tokens  # Document
            + [self.tokenizer.sep_token_id]  # [SEP]
        )
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Add padding if necessary
        padding_length = 512 - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
        
        return input_ids, attention_mask


    def _log_training_data_analysis(self, total_examples: int, positive_examples: int,
                                  negative_examples: int, examples_per_query: List[int]):
        """Log training data analysis results"""
        pos_ratio = positive_examples / total_examples if total_examples > 0 else 0
        
        logger.info("\nTraining Data Analysis:")
        logger.info(f"Total training examples: {total_examples}")
        logger.info(f"Positive examples: {positive_examples} ({pos_ratio:.2%})")
        logger.info(f"Negative examples: {negative_examples} ({1-pos_ratio:.2%})")
        
        if examples_per_query:
            avg_examples = sum(examples_per_query) / len(examples_per_query)
            logger.info(f"Average examples per query: {avg_examples:.1f}")
        
        if pos_ratio < 0.05:
            logger.warning("Very low positive example ratio. Consider increasing n_candidates")
        elif pos_ratio > 0.3:
            logger.info("High positive example ratio. Current n_candidates seems good")
        
        if total_examples < 1000:
            logger.warning("Small training set. Consider increasing n_candidates")

    def train(self, processed_data: List[Dict], initial_retrievals: Dict,
             num_epochs: int = 300, batch_size: int = 64, learning_rate: float = 2e-5,
             patience: int = 30):
        """
        Train the reranker model with early stopping.
        
        Args:
            processed_data: List of processed training examples
            initial_retrievals: Dictionary mapping query IDs to retrieved documents
            num_epochs: Maximum number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            patience: Number of epochs to wait for improvement before early stopping


        """
        
        # Initialize cache and logging managers
        cache_manager = TrainingDataCache()
        
        # Try to load cached examples first
        cache_path = "cache/latest_training_examples.pt"
        if os.path.exists(cache_path):
            try:
                training_examples, metadata = cache_manager.load_examples(cache_path)
                logger.info("Successfully loaded cached training examples")
                max_length = metadata.get('max_length', 512)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                training_examples = None
        else:
            training_examples = None
        
        # Prepare new examples if needed
        if training_examples is None:
            training_examples = self._prepare_training_data(processed_data, initial_retrievals)
            metadata = {
                'max_length': 512,  # Default max length
                'creation_time': datetime.now().isoformat()
            }
            cache_manager.save_examples(training_examples, metadata)
        
        # Start training process
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        if len(training_examples) == 0:
            raise ValueError("No training examples were created")
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            epoch_loss = self._train_epoch(training_examples, optimizer, batch_size)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Average loss: {epoch_loss:.4f}")
            
            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                self.save_model(os.path.join(self.model_save_dir, 'best_model'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
    def _prepare_training_data(self, processed_data: List[Dict], 
                            initial_retrievals: Dict) -> List[Dict]:
        """
        Prepare training examples with proper length handling and smart tokenization.
        
        This method:
        1. Processes each query-document pair
        2. Handles positive/negative examples
        3. Creates properly formatted input tensors
        4. Implements caching for efficiency
        """
        training_examples = []
        logger.info("Preparing training data...")
        
        try:
            for entry in tqdm(processed_data):
                query = entry['query_text']
                correct_laws = set(entry['labels'])
                retrieved_docs = initial_retrievals.get(entry['id'], [])
                
                for doc in retrieved_docs:
                    # Determine if this is a positive example
                    formatted_ref = format_law_reference(doc['file'], doc['index'])
                    is_positive = formatted_ref in correct_laws
                    
                    # Prepare input sequence with proper length handling
                    input_ids, attention_mask = self.prepare_text_pair(query, doc['text'])
                    
                    # Create input tensors
                    inputs = {
                        'input_ids': torch.tensor([input_ids]),
                        'attention_mask': torch.tensor([attention_mask])
                    }
                    
                    # Create label tensor
                    label = torch.tensor([1.0 if is_positive else 0.0])
                    
                    # Store the example
                    training_examples.append({
                        'inputs': inputs,
                        'label': label
                    })
                    
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
        
        logger.info(f"Successfully created {len(training_examples)} training examples")
        return training_examples
    def _train_epoch(self, training_examples: List[Dict], 
                    optimizer: torch.optim.Optimizer,
                    batch_size: int) -> float:
        """Train for one epoch and return average loss"""
        total_loss = 0
        num_batches = 0
        
        for i in tqdm(range(0, len(training_examples), batch_size)):
            batch = training_examples[i:i + batch_size]
            loss = self._train_batch(batch, optimizer)
            
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def _train_batch(self, batch: List[Dict], optimizer: torch.optim.Optimizer) -> float:
        """Train on a single batch with proper padding handling"""
        optimizer.zero_grad()
        
        try:
            # Combine batch inputs (now all tensors should have same length)
            batch_inputs = {
                'input_ids': torch.cat([ex['inputs']['input_ids'] for ex in batch]),
                'attention_mask': torch.cat([ex['inputs']['attention_mask'] for ex in batch])
            }
            
            # Move to device
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
            batch_labels = torch.cat([ex['label'] for ex in batch]).to(self.device)
            
            # Forward pass
            outputs = self.model(**batch_inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                outputs.logits.squeeze(),
                batch_labels
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Error in batch training: {e}")
            # Log batch information for debugging
            logger.error(f"Batch size: {len(batch)}")
            logger.error(f"Input shapes: {[ex['inputs']['input_ids'].shape for ex in batch]}")
            raise

    def save_model(self, save_path: str):
        """Save the model and tokenizer"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
            }, f'{save_path}_model.pt')
            
            self.tokenizer.save_pretrained(f'{save_path}_tokenizer')
            logger.info(f"Model and tokenizer saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

def main():
    """Main execution function"""
    try:
        logger.info("Starting reranker training process")
        
        # Load processed training data
        with open('trainingData/processed_training_data.json', 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        logger.info(f"Loaded {len(processed_data)} training examples")
        
        # Generate initial retrievals
        generator = InitialRetrievalGenerator()
        initial_retrievals = generator.generate_retrievals(processed_data, n_candidates=50)
        
        if not initial_retrievals:
            raise ValueError("No initial retrievals were generated")
        
        # Save initial retrievals
        with open('trainingData/Reranker_retrievals.json', 'w', encoding='utf-8') as f:
            json.dump(initial_retrievals, f, ensure_ascii=False, indent=2)
        logger.info("Saved initial retrievals")
        
        # Initialize and train reranker
        reranker = LegalReranker()
        reranker.analyze_training_data(processed_data, initial_retrievals)
        reranker.train(processed_data, initial_retrievals)
        
        logger.info("Training completed successfully")
    except Exception as e:
            logger.error(f"Error in main execution: {e}")
            raise





if __name__ == "__main__":
    main()

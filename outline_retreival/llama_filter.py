import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, snapshot_download
import re
import logging
from datetime import datetime

def setup_logger():
    """Set up logger with custom format and file handler"""

    
    # Create a logger
    logger = logging.getLogger('LegalAssistant')
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Create file handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(f'outline_retreival/logs/legal_assistant_{timestamp}.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def calculate_max_length(tokenizer, prompt):
    """Calculate appropriate max length based on input prompt length"""
    prompt_tokens = len(tokenizer.encode(prompt))
    response_buffer = 250
    max_length = prompt_tokens + response_buffer
    model_max_length = 22000
    
    logger = logging.getLogger('LegalAssistant')
    logger.debug(f"Prompt length: {prompt_tokens} tokens")
    logger.debug(f"Response buffer: {response_buffer} tokens")
    logger.debug(f"Total max length: {max_length} tokens")
    logger.debug(f"Model max length: {model_max_length} tokens")
    
    final_length = min(max_length, model_max_length)
    logger.debug(f"Final max length: {final_length} tokens")
    return final_length

def clean_response(response, logger):
    """Clean up model response to extract only the law names and scores"""
    logger.debug(f'Original response: {response}')
    try:
        # Find the last occurrence of "請開始分析：" or similar endings
        markers = ["請開始分析：", "分析結果：", "："]
        last_index = -1
        for marker in markers:
            if marker in response:
                last_index = max(last_index, response.rindex(marker))
        
        if last_index != -1:
            result = response[last_index + 1:].strip()
            logger.debug(f"Found marker at index {last_index}")
        else:
            result = response.strip()
            logger.debug("No markers found in response")
            
        logger.debug(f"After marker removal: {result}")
        
        # Remove any numbered list formatting
        result = re.sub(r'^\d+\.\s*', '', result, flags=re.MULTILINE)
        logger.debug(f"After number removal: {result}")
        
        # Split into lines and process each line
        lines = result.split('\n')
        cleaned_parts = []
        
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    law_name = parts[0].strip()
                    score = float(parts[1].strip())
                    cleaned_parts.extend([law_name, str(int(score))])
                    logger.debug(f"Processed law: {law_name} with score: {score}")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to process line: {line}. Error: {e}")
                    continue
        
        final_result = ','.join(cleaned_parts)
        logger.debug(f"Final cleaned response: {final_result}")
        return final_result
    except Exception as e:
        logger.error(f"Error cleaning response: {e}")
        logger.debug("Returning original response due to error")
        return response

def create_prompt(query, document):
    logger = logging.getLogger('LegalAssistant')
    prompt = f"""請分析以下問題與參考法律文件內，每條法律的關聯度並給予分數。

問題：{query}

參考法律文件：
{document}


為每個法律評分(0-100分)，分數代表與問題的相關程度：
    - 90-100分：非常相關，直接針對問題核心
    - 70-89分：相當相關，涵蓋問題重要面向
    - 50-69分：部分相關，涉及問題某些層面
    - 0-49分：關聯性較低

請直接輸出相關法律名稱與分數，格式如下：
法律名稱1,分數1,法律名稱2,分數2
請開始輸出：
"""

    """
    任務：
    1. 參考的法律文件內包含了"法律名稱"與"摘要"，摘要描述了這條法律大致適用的問題與情境，你需要評估每條法律與問題的關聯性，並給予一個評估分數
    2. 為每個法律評分(0-100分)，分數代表與問題的相關程度：
    - 90-100分：非常相關，直接針對問題核心
    - 70-89分：相當相關，涵蓋問題重要面向
    - 50-69分：部分相關，涉及問題某些層面
    - 0-49分：關聯性較低
    3. 不需要考慮問題的真實性，不須考慮是否虛構，不管是不是虛假或虛構的問題，都需要輸出法律名稱與分數
    4. 若文件內的法律都不相關，可以都給予49分以下
    5. 不要回答"這個任務是基於一個虛假的問題和答案，請不要實際應用於真實的情況"
    6. 不要回答參考文件以外的法律名稱

    請直接輸出相關法律名稱與分數，格式如下：
    法律名稱1,分數1,法律名稱2,分數2

    注意事項：
    1. 只輸出法律名稱和分數，不要有其他文字
    2. 分數範圍為0-100
    4. 不要加入編號或其他標記

    請開始輸出：
    """
    
    logger.debug(f"Created prompt with query: {query[:100]}...")  # Log first 100 chars of query
    logger.debug(f"Document length: {len(document)} characters")
    return prompt

def extract_law_names(response):
    """Extract just the law names from the model's response, ignoring scores"""
    logger = logging.getLogger('LegalAssistant')
    logger.debug(f"Extracting law names from response: {response}")
    
    try:
        parts = response.strip().split(',')
        law_names = [parts[i] for i in range(0, len(parts), 2)]
        logger.debug(f"Extracted law names: {law_names}")
        return law_names
    except Exception as e:
        logger.error(f"Error parsing response for law names: {e}")
        logger.debug("Returning empty list due to error")
        return []

def check_gpu(logger):
    """Check GPU availability and log info"""
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Current GPU: {torch.cuda.current_device()}")
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logger.info(f"GPU Memory Usage:")
        logger.info(f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
        logger.info(f"Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB")
    return torch.cuda.is_available()

def load_model(logger):
    logger.info("Starting model loading process")
    login(token="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")
    model_id = "lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.debug("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    logger.debug("Loading model")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    logger.info("Model loading completed")
    return model, tokenizer, device

def generate_response(model, tokenizer, prompt, device, logger):
    max_length = calculate_max_length(tokenizer, prompt)
    logger.debug(f"Prompt tokens: {len(tokenizer.encode(prompt))}")
    logger.debug(f"Setting max_length to: {max_length}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    logger.info('Generating response...')
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.2,  # Add penalty for repetition
            no_repeat_ngram_size=3,  # Prevent repeating 3-grams
            pad_token_id=tokenizer.pad_token_id,
            top_p=1.0
        )
    
    logger.debug('Decoding response')
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_response(response, logger)



def process_queries(input_json_path, output_json_path):
    # Set up logger
    logger = setup_logger()
    logger.info("Starting query processing")
    
    gpu_available = check_gpu(logger)
    model, tokenizer, device = load_model(logger)
    
    logger.info(f"Reading queries from {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    txt_folder = 'outline_retreival/split'
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    logger.info(f"Found {len(txt_files)} text files to process")
    
    output_data = []
    total_queries = len(queries)
    
    for idx, query_item in enumerate(queries, 1):
        query_id = query_item['id']
        query_text = query_item['query_text']
        logger.info(f"Processing query {idx}/{total_queries} (ID: {query_id})")
        
        all_predictions = []
        for txt_file in txt_files:
            logger.debug(f"Processing file: {txt_file}")
            with open(os.path.join(txt_folder, txt_file), 'r', encoding='utf-8') as f:
                current_laws = f.read()
            
            prompt = create_prompt(query_text, current_laws)
            response = generate_response(model, tokenizer, prompt, device, logger)
            law_names = extract_law_names(response)
            
            logger.debug(f"Extracted laws from {txt_file}: {law_names}")
            all_predictions.extend(law_names)
        
        unique_predictions = list(dict.fromkeys(all_predictions))
        output_item = {
            "id": query_id,
            "query_text": query_text,
            "labels": unique_predictions
        }
        output_data.append(output_item)
        logger.info(f"Completed query {idx}/{total_queries}")
    
    logger.info(f"Saving results to {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info("Processing completed successfully")

if __name__ == "__main__":
    input_json_path = "outline_retreival/processed_training_data.json"  
    output_json_path = "outline_retreival/model_predictions.json"  
    process_queries(input_json_path, output_json_path)
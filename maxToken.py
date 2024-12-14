from transformers import AutoTokenizer
import os
from collections import defaultdict

def analyze_token_distribution(directory_path: str, summary: bool = True):
    # Initialize the tokenizer
    model_id = "lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # The instruction template
    instruction = """
    將法律條文轉換為可以匹配日常問題的表示方式。
    需要考慮：
    1. 將專業法律用語對應到一般民眾的日常用語
    2. 識別法條中描述的情境和實際案例的關聯
    3. 保留法條的核心概念和關鍵要素
    4. 連結相似的法律概念和情境
    請轉換以下法條：
    """
    
    # Get instruction token count
    instruction_tokens = len(tokenizer.encode(instruction))
    print(f"Instruction token count: {instruction_tokens}")
    
    # Initialize tracking variables
    max_tokens = 0
    max_token_text = ""
    max_token_file = ""
    token_counts = []
    
    # Create ranges for distribution (0-500, 501-1000, etc.)
    ranges = [(i, i + 499) for i in range(0, 3501, 500)]
    distribution = defaultdict(int)
    
    def parse_line(line: str) -> tuple:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, None
    
    def process_text(text, filename="", index=""):
        nonlocal max_tokens, max_token_text, max_token_file
        
        formatted_text = f"{instruction}\n{text}"
        token_count = len(tokenizer.encode(formatted_text))
        token_counts.append(token_count)
        
        # Update max tokens if needed
        if token_count > max_tokens:
            max_tokens = token_count
            max_token_text = text
            max_token_file = filename
        
        # Update distribution
        for start, end in ranges:
            if start <= token_count <= end:
                distribution[f"{start}-{end}"] += 1
                break
        
        return token_count
    
    total_entries = 0
    
    if summary:
        with open(os.path.join(directory_path), 'r', encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue
                process_text(line.strip(), "summary_file")
                total_entries += 1
    else:
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                with open(file_path, 'r', encoding="utf-8") as file:
                    for line in file.readlines():
                        if not line.strip():
                            continue
                        index, text = parse_line(line)
                        if index and text:
                            process_text(text, filename, index)
                            total_entries += 1
    
    # Calculate statistics
    token_counts.sort()
    median_tokens = token_counts[len(token_counts)//2]
    avg_tokens = sum(token_counts) / len(token_counts)
    
    # Print results
    print("\n=== Token Distribution Analysis ===")
    print(f"\nTotal entries processed: {total_entries}")
    print(f"\nToken count statistics (including instruction):")
    print(f"Maximum: {max_tokens}")
    print(f"Median: {median_tokens}")
    print(f"Average: {avg_tokens:.2f}")
    print(f"Minimum: {token_counts[0]}")
    
    print("\nDistribution by ranges:")
    for range_key in sorted(distribution.keys(), key=lambda x: int(x.split('-')[0])):
        count = distribution[range_key]
        percentage = (count / total_entries) * 100
        print(f"Range {range_key}: {count} entries ({percentage:.2f}%)")
    
    print(f"\nEntry with maximum tokens ({max_tokens}):")
    print(f"File: {max_token_file}")
    print(f"Preview: {max_token_text[:200]}...")
    print(f"\nBreakdown:")
    print(f"Instruction tokens: {instruction_tokens}")
    print(f"Max text-only tokens: {max_tokens - instruction_tokens}")

if __name__ == "__main__":
    directory_path = "lawData/json_DFS"  # Update this path to your data directory
    summary = False  # Set to True or False based on your data format
    analyze_token_distribution(directory_path, summary)
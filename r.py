import re
from typing import List, Tuple

def count_tokens(text: str) -> int:
    """
    Estimate token count in Chinese text.
    This is a simple estimation - each Chinese character and each word separator counts as one token.
    """
    # Count Chinese characters
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # Count non-Chinese words (separated by spaces)
    other_tokens = len(re.findall(r'\b\w+\b', text))
    # Count punctuation and special characters
    punctuation = len(re.findall(r'[^\s\w\u4e00-\u9fff]', text))
    
    return chinese_chars + other_tokens + punctuation

def split_text(text: str, max_tokens: int = 500) -> List[List[str]]:
    """
    Split text into parts where each part has approximately equal token count,
    not exceeding max_tokens, while maintaining row integrity.
    """
    # Split into lines and remove empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Calculate token count for each line
    line_tokens = [(line, count_tokens(line)) for line in lines]
    
    # Initialize variables for splitting
    parts = []
    current_part = []
    current_tokens = 0
    
    # Calculate target tokens per part
    total_tokens = sum(tokens for _, tokens in line_tokens)
    target_parts = (total_tokens + max_tokens - 1) // max_tokens
    target_tokens_per_part = total_tokens / target_parts
    
    for line, tokens in line_tokens:
        # If adding this line would exceed max_tokens and we already have some lines
        if current_tokens + tokens > max_tokens and current_part:
            parts.append(current_part)
            current_part = []
            current_tokens = 0
        
        current_part.append(line)
        current_tokens += tokens
        
        # Check if current part is close to target size
        if current_tokens >= target_tokens_per_part and len(parts) < target_parts - 1:
            parts.append(current_part)
            current_part = []
            current_tokens = 0
    
    # Add the last part if it's not empty
    if current_part:
        parts.append(current_part)
    
    return parts

def write_parts_to_files(parts: List[List[str]], base_filename: str = 'law_part') -> None:
    """Write each part to a separate file."""
    for i, part in enumerate(parts, 1):
        filename = f'{base_filename}_{i}.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(part))
        token_count = sum(count_tokens(line) for line in part)
        print(f'Part {i}: {len(part)} rows, {token_count} tokens - saved to {filename}')

def main():
    try:
        # Read the input file
        with open('section.txt', 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Split the text into parts
        parts = split_text(text, max_tokens=500)
        
        # Write parts to separate files
        write_parts_to_files(parts)
        
        # Print summary
        total_tokens = sum(sum(count_tokens(line) for line in part) for part in parts)
        print(f'\nTotal parts: {len(parts)}')
        print(f'Total tokens: {total_tokens}')
        print(f'Average tokens per part: {total_tokens/len(parts):.2f}')
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
import os
from collections import defaultdict

def process_law_files(directory_path):
    """
    Process text files in the given directory and combine filenames with the same law name,
    preserving the order of sections.
    
    Args:
        directory_path (str): Path to the directory containing the text files
        
    Returns:
        None: Creates a new file with combined law names
    """
    # Dictionary to store law names and their ordered parts
    law_dict = defaultdict(list)
    
    # Get all .txt files in the directory
    txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    
    # Process each file
    for filename in txt_files:
        # Split the filename by spaces (excluding the .txt extension)
        parts = filename[:-4].split(' ')
        
        # The first part is the law name
        law_name = parts[0]
        
        # Add remaining parts to the list for this law name
        remaining_parts = parts[1:]
        
        # For each part in the remaining parts
        for part in remaining_parts:
            # Only add if it's not already in the list
            if part not in law_dict[law_name]:
                law_dict[law_name].append(part)
    
    # Create output file
    output_filename = 'combined_law_names.txt'
    with open(os.path.join(directory_path, output_filename), 'w', encoding='utf-8') as f:
        # Write each law name and its combined parts
        for law_name, parts in law_dict.items():
            # Join parts maintaining order
            combined_parts = ' '.join([law_name] + parts)
            f.write(f'{combined_parts}\n')
    
    print(f'Processing complete. Results written to {output_filename}')

# Example usage
if __name__ == '__main__':
    # Replace with your directory path
    directory_path = 'lawData/json_DFS'  # Current directory
    process_law_files(directory_path)
import json
import os
from pathlib import Path

def process_json_file(json_file_path, output_dir):
    """
    Process a single JSON file and save its hierarchy 5 content to a separate txt file.
    
    Args:
        json_file_path (Path): Path object pointing to the JSON file
        output_dir (Path): Path object pointing to the output directory
        
    The function creates a corresponding txt file in the output directory with the
    same base name as the input JSON file.
    """
    try:
        # Read and parse the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # List to store formatted results for this file
        results = []
        
        def extract_hierarchy_5(node):
            """
            Recursively traverse the JSON tree to find and extract hierarchy 5 nodes.
            
            Args:
                node (dict): Current node in the JSON tree
                
            This helper function adds formatted strings to the results list whenever
            it finds a hierarchy 5 node.
            """
            # Check if current node is hierarchy 5
            if node.get('hierarchy') == 5:
                # Remove '第' and '條' from the number and clean up whitespace
                number = node['number'].replace('第', '').replace('條', '').strip()
                
                # Join all content lines with spaces to create a single line
                content = ' '.join(node['content'])
                
                # Create the formatted output string and add to results
                results.append(f"{number} {content}")
            
            # Continue searching in all child nodes
            for child in node.get('children', []):
                extract_hierarchy_5(child)
        
        # Start the recursive extraction process
        extract_hierarchy_5(data)
        
        if results:
            # Create output filename by replacing .json extension with .txt
            output_filename = json_file_path.stem + '.txt'
            output_path = output_dir / output_filename
            
            # Write the results to the output file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(results))
            
            print(f"Successfully processed {json_file_path.name} -> {output_filename}")
        else:
            print(f"No hierarchy 5 content found in {json_file_path.name}")
            
    except Exception as e:
        print(f"Error processing {json_file_path.name}: {str(e)}")

def process_directory(input_dir):
    """
    Process all JSON files in the input directory and save results to a new folder.
    
    Args:
        input_dir (str): Path to the directory containing JSON files
        
    This function creates a new 'extracted_content' folder in the input directory
    to store the output txt files.
    """
    # Convert input directory string to Path object
    input_path = Path(input_dir)
    
    # Create output directory path
    output_dir = input_path / 'extracted_content'
    
    # Create the output directory if it doesn't exist
    try:
        output_dir.mkdir(exist_ok=True)
        print(f"Output directory created/confirmed at: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory: {str(e)}")
        return
    
    # Find and process all JSON files
    json_files = list(input_path.glob('*.json'))
    
    if not json_files:
        print("No JSON files found in the input directory")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each JSON file
    for json_file in json_files:
        process_json_file(json_file, output_dir)
    
    print("\nProcessing complete. Check the 'extracted_content' folder for results")

# Usage example
if __name__ == "__main__":
    # Replace with your directory path
    directory_path = 'law_unpress/processed_json'
    process_directory(directory_path)














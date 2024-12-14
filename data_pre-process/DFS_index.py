import os
import re
from typing import List

class TextProcessor:
    def __init__(self, input_folder: str, output_folder: str):
        self.input_folder = input_folder
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output directory: {output_folder}")

    def process_line(self, line: str) -> List[str]:
        """Process a line of text and extract article numbers."""
        # Split by comma
        parts = line.split(',')
        results = []
        
        for part in parts:
            # Extract the number before "條"
            match = re.search(r'(\d+(?:-\d+)?)\s*條', part.strip())
            if match:
                number = match.group(1)
                # Validate number format
                try:
                    number_parts = number.split('-')
                    valid = all(0 <= int(num) <= 9999 for num in number_parts)
                    if valid:
                        results.append(number)
                except ValueError:
                    continue
        
        return results

    def process_file(self, input_path: str, output_path: str):
        """Process a single file and write the extracted numbers to output file."""
        try:
            print(f"\nProcessing file: {input_path}")
            
            # Read the input file
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Process content
            numbers = self.process_line(content)
            
            if numbers:
                print(f"Found {len(numbers)} valid article numbers")
                print(f"Writing to: {output_path}")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(','.join(numbers))
                
                print(f"Successfully wrote to {output_path}")
                print("Extracted numbers:", ','.join(numbers))
            else:
                print(f"No valid article numbers found in {input_path}")
                    
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")

    def process_files(self):
        """Process all files in the input folder."""
        print(f"\nStarting processing of files in: {self.input_folder}")
        file_count = 0
        
        for filename in os.listdir(self.input_folder):
            if filename.endswith('.txt'):
                input_path = os.path.join(self.input_folder, filename)
                output_path = os.path.join(self.output_folder, filename)
                
                print(f"\nProcessing file {file_count + 1}: {filename}")
                self.process_file(input_path, output_path)
                file_count += 1
        
        print(f"\nProcessed {file_count} files")

def main():
    input_folder = "../lawData/json_DFS"  # Folder containing original text files
    output_folder = "../lawData/json_DFS_index"  # Folder for processed output files
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    # Verify input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} does not exist!")
        return
        
    processor = TextProcessor(input_folder, output_folder)
    processor.process_files()

if __name__ == "__main__":
    main()







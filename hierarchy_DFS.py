import json
import os
import re
from typing import List, Dict, Any, Tuple
import string
class LawProcessor:
    def __init__(self, input_folder: str, output_folder: str, txt_folder: str):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.txt_folder = txt_folder
        self.current_law_content = {}  # Cache for current law's txt content
        self.current_file_name = ""    # Add this to track current file name
        os.makedirs(output_folder, exist_ok=True)

    def extract_law_name(self, content: List[str]) -> str:
        for item in content:
            if item.startswith("法規名稱："):
                cleaned_item = item.replace("法規名稱：", "").strip()
                return cleaned_item



    def extract_law_name_summary(self, content: List[str]) -> str:
        """Extract pure law name from content"""
        cleaned_content = []
        for item in content:
            if item.startswith("法規名稱："):
                cleaned_item = item.replace("法規名稱：", "").strip()
                cleaned_content.append(cleaned_item)
            elif item.startswith("摘要："):
                cleaned_item = item.replace("摘要：", "").strip()
                cleaned_content.append(cleaned_item)
            else:
                cleaned_content.append(item)
        
        # Join the content first
        joined_content = ", ".join(cleaned_content)
        
        # Define Chinese punctuation to replace
        chinese_punct = "。，、；：「」『』（）［］【】《》〈〉〔〕・"
        
        # Create translation table
        trans_table = str.maketrans({char: "," for char in chinese_punct})
        
        # Replace Chinese punctuation with comma
        result = joined_content.translate(trans_table)
        
        # Clean up multiple consecutive commas
        result = re.sub(r',+', ',', result)
        # Clean up spaces around commas
        result = re.sub(r'\s*,\s*', ', ', result)
        # Remove leading/trailing commas and whitespace
        result = result.strip(' ,')
        
        return result
    def load_txt_content(self, law_name: str) -> Dict[str, str]:
        """Load and parse txt file content"""
        result = {}
        txt_path = os.path.join(self.txt_folder, f"{law_name}.txt")
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.read().strip().split('\n')
                for line in lines:
                    # Split by first space to separate index and content
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        index, content = parts
                        result[index] = content
        except FileNotFoundError:
            print(f"Warning: {txt_path} not found")
        return result

    def convert_chinese_number_to_index(self, number: str) -> str:
        """Convert Chinese style article number to index format"""
        # Remove "第" and "條"
        number = number.replace("第", "").replace("條", "").strip()
        
        # Convert Chinese numbers to Arabic numbers if needed
        chinese_nums = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
                       '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'}
        
        # Handle special case for numbers with "之"
        if "之" in number:
            main_num, sub_num = number.split("之")
            main_num = ''.join(chinese_nums.get(c, c) for c in main_num)
            sub_num = ''.join(chinese_nums.get(c, c) for c in sub_num)
            return f"{main_num}-{sub_num}"
        
        # Convert regular numbers
        result = ''.join(chinese_nums.get(c, c) for c in number)
        return result.strip()

    def clean_filename(self, content: str) -> str:
        cleaned = re.sub(r'[^\w\s]', ' ', content)
        return ' '.join(cleaned.split())

    def should_skip_number(self, number: str, hierarchy: int) -> bool:
        # Clean up the input string
        number = number.strip()
        
        # Match pattern "第 n 條" or "第 n-m 條"
        match = re.search(r'第\s*(\d+(?:-\d+)?)\s*條', number)
        if not match:
            return True
            
        # Validate number format
        try:
            number_parts = match.group(1).split('-')
            return not all(0 <= int(num) <= 9999 for num in number_parts)
        except ValueError:
            return True

    def extract_content(self, node: Dict[Any, Any]) -> List[str]:
        contents = []
        hierarchy = node.get('hierarchy', 0)
        
        if hierarchy == 0:
            # For hierarchy 0, extract law name and summary
            if node.get('content'):
                law_name = self.extract_law_name_summary(node['content'])
                if law_name:
                    contents.append(law_name)
        else:
            number = node.get('number', '')
            if not self.should_skip_number(number, hierarchy):
                if number:
                    contents.append(number)
            if node.get('content'):
                contents.extend(node['content'])
        
        return contents

    def update_hierarchy5_content(self, node: Dict[Any, Any]) -> List[str]:
        """Update and collect content from hierarchy 5 nodes using txt file content"""
        contents = []
        if node.get('hierarchy') == 5:
            number = node.get('number', '')
            if number:
                # Check if original content is "（刪除）"
                original_content = node.get('content', [])
                if original_content and original_content[0] == "（刪除）":
                    return contents  # Skip this number and content
                
                contents.append(number)
                # Convert number to index format
                index = self.convert_chinese_number_to_index(number)
                # If we have matching content in txt file, use it instead of original content
                if index in self.current_law_content:
                    contents.append(self.current_law_content[index])
                elif node.get('content'):  # Fallback to original content if no match
                    print(f"Warning: No matching content found in {self.current_file_name} for number: {number} (index: {index})")
                    contents.extend(node['content'])
        return contents

    def dfs(self, node: Dict[Any, Any], path_contents: List[str], all_paths: List[Tuple[List[str], List[str]]]):
        current_hierarchy = node.get('hierarchy', 0)
        
        # Add current node's contents to the path
        if current_hierarchy < 5:
            current_contents = self.extract_content(node)
            path_contents.extend(current_contents)
        
        children = node.get('children', [])
        if not children and current_hierarchy == 5:
            # We've reached a hierarchy 5 node
            level5_contents = self.update_hierarchy5_content(node)
            if level5_contents:
                all_paths.append((path_contents.copy(), level5_contents))
        else:
            for child in node.get('children', []):
                self.dfs(child, path_contents.copy(), all_paths)

    def process_files(self):
        for filename in os.listdir(self.input_folder):
            if filename.endswith('.json'):
                self.current_file_name = filename  # Update the current file name
                file_path = os.path.join(self.input_folder, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract law name and load corresponding txt content
                    if data.get('content'):
                        law_name = self.extract_law_name(data['content'])
                        self.current_law_content = self.load_txt_content(law_name)
                    
                    # Collect all paths through DFS
                    all_paths = []
                    self.dfs(data, [], all_paths)
                    
                    # Group paths by their non-hierarchy-5 content
                    path_groups = {}
                    for path_content, level5_content in all_paths:
                        path_key = tuple(path_content)
                        if path_key not in path_groups:
                            path_groups[path_key] = []
                        path_groups[path_key].extend(level5_content)
                    
                    # Create output files for each unique path
                    for path_content, level5_contents in path_groups.items():
                        filename = self.clean_filename(' '.join(path_content))
                        if len(filename) > 150:
                            filename = filename[:150]
                        
                        all_contents = list(path_content) + level5_contents
                        output_path = os.path.join(self.output_folder, f"{filename}.txt")
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(', '.join(all_contents))
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

def main():
    # Configure input and output folders
    input_folder = "lawData/law_unpress/processed_json"  # Change this to your input folder path
    output_folder = "lawData/json_DFS"  # Change this to your output folder path
    txt_folder = "lawData/law_KeywordExtracted"  # Folder containing the .txt files
    
    processor = LawProcessor(input_folder, output_folder, txt_folder)
    processor.process_files()

if __name__ == "__main__":
    main()



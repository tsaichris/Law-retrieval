import pandas as pd
from striprtf.striprtf import rtf_to_text
import re
import json
import os
from pathlib import Path

path = 'law_unpress' # folder

# Let's modify the main processing logic to handle multiple files
def process_rtf_files(folder_path):
    # Create a Path object for better cross-platform compatibility
    directory = Path(folder_path)
    
    # Create an output directory for JSON files if it doesn't exist
    output_dir = directory / 'processed_json'
    output_dir.mkdir(exist_ok=True)
    
    # Iterate through all RTF files in the directory
    for rtf_file in directory.glob('*.rtf'):
        try:
            # Extract text from the RTF file
            text = extract_text_from_rtf(str(rtf_file))
            
            # Process the text using your existing hierarchical_chunk3 function
            output = hierarchical_chunk3(text)
            
            # Create output filename - replace .rtf with .json
            output_filename = output_dir / f"{rtf_file.stem}.json"
            
            # Save the processed data as JSON
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=4)
                
            print(f"Successfully processed {rtf_file.name}")
            
        except Exception as e:
            print(f"Error processing {rtf_file.name}: {str(e)}")


def extract_text_from_rtf(input_file):
    with open(input_file, 'r') as file:
        text = rtf_to_text(file.read())
    return text

def chinese_to_arabic_numerals(text):
    chinese_digits = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10, '百': 100, '千': 1000}
    result = 0
    umit = 0
    for char in reversed(text):
        if char in chinese_digits:
            digit = chinese_digits[char]
            if digit >= 10:
                if digit > umit:
                    umit = digit
                else:
                    umit *= digit
            else:
                result += umit * digit
    return result

def hierarchical_chunk3(raw_text):
    part_pattern = re.compile(r"^\s*(第\s*[一二三四五六七八九十百千萬]+\s*編\s*)")
    chapter_pattern = re.compile(r"^\s*(第\s*[一二三四五六七八九十百千萬]+\s*章\s*)")
    section_pattern = re.compile(r"^\s*(第\s*[一二三四五六七八九十百千萬]+\s*節\s*)")
    article_pattern = re.compile(r"^\s*(第\s*[一二三四五六七八九十百千萬]+\s*款\s*)")
    paragraph_pattern = re.compile(r"^\s*(第\s*\d+(-\d+)?\s*條\s*[^\n]*)") 

    hierarchy = {"hierarchy": 0, 'content': [], "children": []}
    current_levels = [hierarchy]

    def ensure_parent_levels(required_level):
        while len(current_levels) <= required_level:
            parent_level = len(current_levels)
            placeholder = {
                "hierarchy": parent_level,
                "number": f"without_level{parent_level}",
                "content": [],
                "children": []
            }
            if parent_level > 1:
                current_levels[parent_level-1]["children"].append(placeholder)
            current_levels.append(placeholder)

    def ensure_root_exists():
        if len(hierarchy["children"]) == 0:
            default_root = {
                "hierarchy": 1,
                "number": "Without_level1",
                "content": [],
                "children": []
            }
            hierarchy["children"].append(default_root)
            current_levels.append(default_root)

    lines = raw_text.splitlines()
    for line in lines:

        if not line.strip():
            continue

        if part_match := part_pattern.match(line):
            number = f"{part_match.group(1)}"
            new_node = {
                "hierarchy": 1,
                "number": number,
                "content": [line.strip().replace(number, '').strip()],
                "children": []
            }
            current_levels = [hierarchy, new_node]
            hierarchy["children"].append(new_node)
        elif chapter_match := chapter_pattern.match(line):
            ensure_root_exists()  
            ensure_parent_levels(1)
            number = f"{chapter_match.group(1)}"
            new_node = {
                "hierarchy": 2,
                "number": number,
                "content": [line.strip().replace(number, '').strip()],
                "children": []
            }
            current_levels = current_levels[:2]
            current_levels.append(new_node)
            current_levels[1]["children"].append(new_node)
        elif section_match := section_pattern.match(line):
            ensure_root_exists()  
            ensure_parent_levels(2)
            number = f"{section_match.group(1)}"
            new_node = {
                "hierarchy": 3,
                "number": number,
                "content": [line.strip().replace(number, '').strip()],
                "children": []
            }
            current_levels = current_levels[:3]
            current_levels.append(new_node)
            current_levels[2]["children"].append(new_node)
        elif article_match := article_pattern.match(line):
            ensure_root_exists()  
            ensure_parent_levels(3)
            number = f"{article_match.group(1)}"
            new_node = {
                "hierarchy": 4,
                "number": number,
                "content": [line.strip().replace(number, '').strip()],
                "children": []
            }
            current_levels = current_levels[:4]
            current_levels.append(new_node)
            current_levels[3]["children"].append(new_node)
        elif paragraph_match := paragraph_pattern.match(line):
            ensure_root_exists() 
            ensure_parent_levels(4)
            number = f"{paragraph_match.group(1)}"
            new_node = {
                "hierarchy": 5,
                "number": number,
                "content": [],  
                "children": []
            }
            current_levels = current_levels[:5]
            current_levels.append(new_node)
            current_levels[4]["children"].append(new_node)
        else:
            current_levels[-1]["content"].append(line.strip())

    ensure_root_exists() 

    return hierarchy




if __name__ == "__main__":
    # Specify your input folder path
    input_folder = "law_unpress"  # or use absolute path if needed
    process_rtf_files(input_folder)
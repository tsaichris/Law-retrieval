import os

def process_file(file_path: str):
    #print(f"\nProcessing file: {os.path.basename(file_path)}")
    
    # Track duplicates within this file
    seen_indices = {}  # index -> line numbers
    seen_lines = {}    # line content -> line numbers
    modified_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # Track duplicates
            if ' ' in line:
                index = line.split(' ', 1)[0]
                if index not in seen_indices:
                    seen_indices[index] = []
                seen_indices[index].append(line_num)
                
                if line not in seen_lines:
                    seen_lines[line] = []
                seen_lines[line].append(line_num)
            
            # Fix brackets
            line = line.replace('[', '').replace(']', '')
            
            # Fix spacing
            if ' ' not in line:
                # Find where the index ends and content begins
                for i, char in enumerate(line):
                    if not char.isdigit() and char != '-':
                        line = line[:i] + ' ' + line[i:]
                        break
            
            modified_lines.append(line)
    
    # Print duplicate information for this file
    has_duplicates = False
    
    # Check for duplicate indices
    for index, line_nums in seen_indices.items():
        if len(line_nums) > 1:
            if not has_duplicates:
                print(f"\nDuplicates in {os.path.basename(file_path)}:")
                print("=" * 80)
                has_duplicates = True
            print(f"\nDuplicate index '{index}' found at lines: {line_nums}")
    
    # Check for duplicate lines
    for line, line_nums in seen_lines.items():
        if len(line_nums) > 1:
            if not has_duplicates:
                print(f"\nDuplicates in {os.path.basename(file_path)}:")
                print("=" * 80)
                has_duplicates = True
            print(f"\nDuplicate line found at lines {line_nums}:")
            print(f"Content: {line}")
    
    if not has_duplicates:
        #print("No duplicates found.")
        pass
    
    # Write back the modified file
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in modified_lines:
            file.write(line + '\n')

def main():
    directory_path = 'data'
    
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist!")
        return
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            process_file(file_path)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
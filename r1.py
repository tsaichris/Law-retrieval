def process_text_file(input_text):
    # Split the text into lines
    lines = input_text.split('\n')
    
    # Step 1: Add prefix and insert "摘要:" after the first segment
    final_lines = []
    for line in lines:
        if not line.strip():  # Skip empty lines
            continue
            
        # Split the line by first space
        parts = line.split(' ', 1)
        if len(parts) == 2:  # If there's at least one space in the line
            law_name, description = parts
            processed_line = f"法律名稱:{law_name} 摘要:{description}"
        else:  # If there's no space, just add the prefix
            processed_line = f"法律名稱:{line}"
        final_lines.append(processed_line)
    
    # Step 2: Sort lines by length in descending order
    line_lengths = [(len(line), line) for line in final_lines if line.strip()]
    sorted_lines = [line for _, line in sorted(line_lengths, key=lambda x: x[0], reverse=True)]
    
    return "\n".join(sorted_lines)

# Example usage:
if __name__ == "__main__":
    # Read the input file
    try:
        with open('section.txt', 'r', encoding='utf-8') as file:
            input_text = file.read()
            
        # Process the text
        result = process_text_file(input_text)
        
        # Write the result to a new file
        with open('processed_law.txt', 'w', encoding='utf-8') as file:
            file.write(result)
            
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
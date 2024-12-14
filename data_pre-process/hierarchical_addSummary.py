import json
import os
from typing import Dict, Any

def read_json_file(file_path: str) -> Dict[str, Any]:
    """Read and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None

def write_json_file(file_path: str, data: Dict[str, Any]) -> bool:
    """Write data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"Error writing to {file_path}: {str(e)}")
        return False

def update_law_files():
    """Main function to update law files with summaries."""
    # Read the summary file
    summary_data = read_json_file('summary.json')
    if not summary_data:
        return

    # Create a mapping of law names to their summaries
    summary_map = {item['名稱']: item['內容'] for item in summary_data}

    # Get all JSON files in the unpress directory
    allLaw_json = '../lawData/law_unpress/processed_json'
    json_files = [f for f in os.listdir(allLaw_json) if f.endswith('.json')]

    # Process each law file
    for json_file in json_files:
        try:
            # Extract law name from filename (removing .json extension)
            law_name = os.path.splitext(json_file)[0]
            
            # Check if we have a summary for this law
            if law_name not in summary_map:
                print(f"No summary found for {law_name}")
                continue


            # Read the law file with full path
            law_file_path = os.path.join(allLaw_json, json_file)
            law_data = read_json_file(law_file_path)
            if not law_data:
                continue

            # Add or update the summary in the content array at hierarchy 0
            if law_data.get('hierarchy') == 0:
                #if not isinstance(law_data.get('content'), list):
                    #law_data['content'] = []

                # Keep only the first line of original content
                if law_data['content']:
                    first_line = law_data['content'][0]
                    law_data['content'] = [
                        first_line,
                        f"摘要：{summary_map[law_name]}"
                    ]
                #print(law_data['content'])

            # Write the updated content back to the file
            if write_json_file(law_file_path, law_data):
                #print(f"Successfully updated {json_file}")
                pass
            else:
                #print(f"Failed to update {json_file}")
                pass

        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")

if __name__ == "__main__":
    update_law_files()
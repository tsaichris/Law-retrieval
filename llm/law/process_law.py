import os
import json
import re
from striprtf.striprtf import rtf_to_text
from spire.doc import *

def read_rtf_striprtf(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            rtf_content = file.read()
            text = rtf_to_text(rtf_content)
            return text.split('\n')
    except Exception as e:
        print(f"使用striprtf讀取失敗: {str(e)}")
        return None

def read_rtf_spire(file_path):
    try:
        document = Document()
        document.LoadFromFile(file_path)
        temp_docx = "temp.docx"
        document.SaveToFile(temp_docx, FileFormat.Docx2019)
        document.Close()
        
        # 讀取轉換後的docx檔案
        doc = Document(temp_docx)
        content = [para.text for para in doc.paragraphs]
        
        # 刪除暫存檔
        os.remove(temp_docx)
        return content
    except Exception as e:
        print(f"使用Spire.Doc讀取失敗: {str(e)}")
        return None

def extract_law_content(file_path, law_name):
    # 嘗試使用striprtf讀取
    content_lines = read_rtf_striprtf(file_path)
    
    # 如果striprtf失敗，嘗試使用Spire.Doc
    if content_lines is None:
        content_lines = read_rtf_spire(file_path)
        if content_lines is None:
            raise Exception("無法讀取RTF檔案")

    law_dict = {}
    current_article = ""
    current_content = []
    
    for text in content_lines:
        text = text.strip()
        if not text:
            continue
            
        article_match = re.match(r'^第\s*(\d+)(?:-(\d+))?\s*條', text)
        
        if article_match:
            if current_article and current_content:
                law_dict[current_article] = ' '.join(current_content)
            
            main_num = article_match.group(1)
            sub_num = article_match.group(2)
            
            if sub_num:
                current_article = f"{law_name}第{main_num}之{sub_num}條"
            else:
                current_article = f"{law_name}第{main_num}條"
                
            current_content = [text]
        else:
            if current_article:
                current_content.append(text)
    
    if current_article and current_content:
        law_dict[current_article] = ' '.join(current_content)
    
    return law_dict

def save_to_json(law_dict, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(law_dict, f, ensure_ascii=False, indent=4)

def main():
    rtf_files = [f for f in os.listdir('.') if f.endswith('.rtf')]
    
    for rtf_file in rtf_files:
        try:
            law_name = re.sub(r'^\d+', '', rtf_file[:-4])
            print(f"正在處理 {rtf_file}...")
            
            law_dict = extract_law_content(rtf_file, law_name)
            json_file = rtf_file.replace('.rtf', '.json')
            save_to_json(law_dict, json_file)
            
            print(f"已成功處理 {rtf_file} 並儲存為 {json_file}")
            print("\n範例內容：")
            for key, value in list(law_dict.items())[:2]:
                print(f"\nKey: {key}")
                print(f"Value: {value}")
                
        except Exception as e:
            print(f"處理 {rtf_file} 時發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()
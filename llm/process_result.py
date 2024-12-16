import pandas as pd
import re

# 讀取 CSV 檔案
data = pd.read_csv('result.csv')

# 清除特殊字元的函數
def clean_special_characters(target):
    if pd.isna(target):
        return " "
    # 移除特殊字元和問號
    cleaned_target = re.sub(r'[|<>�]', '', target)
    return cleaned_target



pattern = r'(?:\d+\.)?([\u4e00-\u9fa5]+第\d+條(?:之\d+)?)'

def remove_spaces(law_string):
    return re.sub(r'(第)\s*(\d+條)', r'\1\2', law_string)

def clean_target_format(target):
    if pd.isna(target):
        return " "
    # 先移除法條中的空白
    target = remove_spaces(target)
    # 處理換行和其他空白
    target = target.replace('\n', '')
    laws = target.split(',')
    valid_laws = set()
    for law in laws:
        print(law)
        law = law.strip()
        law = law.replace('\\', '').strip()
        matches = re.findall(pattern, law)
        valid_laws.update(matches)
    return ','.join(valid_laws) if valid_laws else " "



# 先清除特殊字元，再進行格式檢查
data['TARGET'] = data['TARGET'].apply(clean_special_characters)
data['TARGET'] = data['TARGET'].apply(clean_target_format)

# 儲存處理後的資料
data.to_csv('cleaned_result.csv', 
            index=False, 
            quoting=1,           # 使用引號
            quotechar='"',       # 指定引號字符
            columns=['id', 'TARGET'],  # 指定列順序
            encoding='utf-8',
            doublequote=True)    # 處理引號內的引號

print("\n已將處理後的資料儲存為 'cleaned_result.csv'")
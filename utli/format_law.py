

def format_law_reference(file_name: str, index: str) -> str:
    """
    Convert file name and index into standardized law reference format.
    
    Args:
        file_name: Name of the law file (e.g., "民法.txt")
        index: Article index, may include dash for sub-articles (e.g., "444" or "426-1")
    
    Returns:
        Formatted law reference following the standard convention
    
    Examples:
        >>> format_law_reference("民法.txt", "444")
        "民法第444條"
        >>> format_law_reference("民法.txt", "426-1")
        "民法第426條之1"
    """
    try:
        # Remove .txt extension and any whitespace
        base_name = file_name.replace('.txt', '').strip()
        
        # Handle cases where index contains a dash (indicating sub-article)
        if '-' in str(index):
            main_number, sub_number = str(index).split('-')
            return f"{base_name}第{main_number.strip()}條之{sub_number.strip()}"
        else:
            return f"{base_name}第{str(index).strip()}條"
    except Exception as e:
        print(f"Error formatting law reference for {file_name} {index}: {e}")
        return ""
    


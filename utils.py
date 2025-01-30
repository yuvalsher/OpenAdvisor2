##############################################################################
import json
import os


def flip_by_line(text: str) -> str:
    """Flip the text from Hebrew to English and vice versa"""
    lines = text.split('\n')
    flipped_lines = []
    for line in lines:
        # Count Hebrew characters (Unicode range for Hebrew: 0x0590-0x05FF)
        hebrew_count = sum(1 for c in line if '\u0590' <= c <= '\u05FF' or c == ' ')
        if hebrew_count > len(line)/4:
            flipped_lines.append(line[::-1])
        else:
            flipped_lines.append(line)
    text = '\n'.join(flipped_lines)
    return text

##############################################################################
def load_json_file(file_name: str, config: dict):
    # Use the DB_Path from the config
    full_path = os.path.join(config["DB_Path"], file_name)
    if not os.path.exists(full_path):
        print(f"File {full_path} does not exist.")
        raise FileNotFoundError(f"File {full_path} does not exist.")
    
    try:    
        with open(full_path, "r", encoding='utf-8') as f:
            # read file contents into string    
            file_contents = f.read()
            # Remove BOM and direction marks
            file_contents = file_contents.strip('\ufeff\u200e\u200f')
            return json.loads(file_contents)
    except Exception as e:
        print(f"Error loading JSON file {file_name}: {str(e)}")
        raise e


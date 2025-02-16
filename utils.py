import json
import os
from openai import AsyncOpenAI, OpenAI
import requests
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import logging

# Set the logging verbosity to ERROR to suppress warnings
logging.set_verbosity_error()

##############################################################################
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

##############################################################################
def get_hebert_embedding(text, max_length: int = 5000):
    """
    Generates an embedding for the given Hebrew text using the HeBERT model.

    Args:
        text (str): The Hebrew text to be embedded.

    Returns:
        torch.Tensor: The embedding vector for the input text.
    """
    # Load the HeBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
    model = AutoModel.from_pretrained("avichr/heBERT")

    # Tokenize the input text and convert to PyTorch tensors
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

    # Generate the embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the embeddings from the model output
    # outputs.last_hidden_state contains embeddings for all tokens
    # We can average them to get a single vector for the entire input text
    embeddings = outputs.last_hidden_state.mean(dim=1)

    # Convert the tensor to a list for JSON serialization
    embedding_list = embeddings.squeeze().tolist()

    return embedding_list

##############################################################################
def get_longhero_embedding(text):
    """
    Generates an embedding for the given Hebrew text using the LongHeRo model.

    Args:
        text (str): The Hebrew text to be embedded.

    Returns:
        np.ndarray: The embedding vector for the input text as a NumPy array.
    """
    # Load the LongHeRo tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('HeNLP/LongHeRo')
    model = AutoModel.from_pretrained('HeNLP/LongHeRo')

    # Tokenize the input text and convert to PyTorch tensors
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096  # LongHeRo supports sequences up to 4096 tokens
    )

    # Generate the embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the embeddings from the model output
    # outputs.last_hidden_state contains embeddings for all tokens
    # We can average them to get a single vector for the entire input text
    embeddings = outputs.last_hidden_state.mean(dim=1)

    # Convert the tensor to a list of floats
    embedding_list = embeddings.squeeze().tolist()

    return embedding_list

##############################################################################
def extract_html_body(full_html: str):
    error_404 = """<html><head>
		<!-- Error 404 .-->
		<meta http-equiv="Content-Type" content="text/html; charset=windows-1255">
		<title>האוניברסיטה הפתוחה</title>

		</head><html><head>
		<!-- Error 404 .-->
		<meta http-equiv="Content-Type" content="text/html; charset=windows-1255">
		<title>האוניברסיטה הפתוחה</title>

		</head>"""
    book_page = """<html itemscope="" itemtype="http://schema.org/Book" class=" supports csstransforms3d" lang="en"><head>"""

    start_markers = [
        "<!--end header-->",
        "<div id=\"content\">",
        "<!-- end menu cellular -->",
        "<div id=\"www_widelayout\">",
        "<div class=\"main-content maincontentplaceholder\">",
        "<!--start content -->",
        "<!-- BEGIN ZONES CONTAINER -->",
        "<!--תוכן-->"
    ]
    end_markers = [
        "<!--footer-->",
        "<!-- footer -->",
        "<!--end content -->",
        "<!-- END ZONES CONTAINER -->",
        "<!--END תוכן-->"
    ]

    if full_html.startswith(error_404):
        return None
    if full_html.startswith(book_page):
        return None
    
    lhtml = full_html.lower()
    start_locs = list(map(lambda marker: lhtml.find(marker), start_markers))
    start_loc = max(start_locs)
    end_locs = list(map(lambda marker: lhtml.find(marker), end_markers))
    end_loc = max(end_locs)
    if start_loc == -1 or end_loc == -1 or start_loc > end_loc:
        return None
    
    extracted_content = full_html[start_loc:end_loc].strip()
    return extracted_content

##############################################################################
# Function to escape special characters
def escape_content(content):
    return content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    
##############################################################################
def unesacape_content(content):
    content = content.replace('\\n', '\n')
    content = content.replace('\\"', '"')
    content = content.replace('\\\\', '\\')
    return content

##############################################################################
async def get_html_from_url(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an exception for HTTP errors
        full_html_content = response.text
    except requests.exceptions.RequestException as error:
        print(f"Error fetching URL {url}: {error}")
        full_html_content = ""
    return full_html_content

##############################################################################
system_instructions = """
Convert the following HTML content into a concise markdown document.
Ignore the following html elements:
    - top menues, side menues
    - header, footer
    - ads
    - social media links
    - images
    - videos
    - audio
    - other non-text elements
"""

##############################################################################
async def get_md_from_html(html: str, openai_client: AsyncOpenAI) -> str:
    html_content = extract_html_body(html)
    if not html_content:
        return None
    
    escaped_html = escape_content(html_content)
    system = escape_content(system_instructions)

    try:
        completion = await openai_client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal::B08ve6Iy",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": escaped_html}
            ]
        )
    except Exception as error:
        print("Failed to create chat completion:", error)
        return

    md = completion.choices[0].message.content
    md = unesacape_content(md)
    return md
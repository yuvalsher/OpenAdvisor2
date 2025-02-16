from datetime import datetime
import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import html2text
import json
import tiktoken # for token counting
import numpy as np
from collections import defaultdict
from openai import OpenAI

from utils import escape_content, unesacape_content, extract_html_body

openai_client = OpenAI()

# List of URLs
urls = [
    "https://academic.openu.ac.il/cs/bsc_excellence/pages/programs.aspx",
    "https://academic.openu.ac.il/cs/bsc_excellence/pages/interested.aspx",
    "http://www.openu.ac.il/dean-students/scholarships/pages/FinancialAidScholarship.aspx",
    "http://academic.openu.ac.il/education_psychology/social-psychology/Pages/about.aspx",
    "http://academic.openu.ac.il/education_psychology/social-psychology/program/m07.aspx",
    "http://academic.openu.ac.il/education_psychology/social-psychology/Pages/staff.aspx",
    "http://academic.openu.ac.il/education_psychology/social-psychology/pages/specialization.aspx",
#    "http://www.openu.ac.il/academic/support/firstcourse.html", Too big
    "https://academic.openu.ac.il/english/pages/default.aspx",
    "http://www.openu.ac.il/overseas/index.html",
    "http://www.openu.ac.il/en/research/eu/pages/default.aspx",
    "http://www.openu.ac.il/research/news/stanford2.aspx",
    "http://academic.openu.ac.il/cs/mlbd/pages/default.aspx",
    "http://www.openu.ac.il/deanacademicstudies/pages/soldiers.aspx",
    "http://www.openu.ac.il/young/pages/math_5_points.aspx",
    "http://www.openu.ac.il/deanacademicstudies/teachandlearn/learning_skiils/pages/tipim_meida.aspx",
    "http://www.openu.ac.il/deanacademicstudies/teachandlearn/learning_skiils/pages/zevet.aspx",
    "http://academic.openu.ac.il/cs/computer/program/l17.aspx",
    "http://www.openu.ac.il/about/milestones/pages/default.aspx",
    "http://academic.openu.ac.il/more-than-degree/Pages/default.aspx",
    "http://www.openu.ac.il/library/Pages/article_book_straight_home.aspx",
    "http://www.openu.ac.il/dean-students/opjob/practice/pages/chemistry.aspx",
    "http://www.openu.ac.il/deanacademicstudies/pages/disciplinecommittee.aspx",
    "http://www.openu.ac.il/deanacademicstudies/pages/conservation.aspx",
    "http://www.openu.ac.il/en/research/eu/funding_opportunities/pages/other_opportunities.aspx",
    "http://academic.openu.ac.il/Education_Psychology/Cognition/program/G222.aspx",
    "http://academic.openu.ac.il/History/ppe/Pages/default.aspx",
    "http://academic.openu.ac.il/economics/mba/pages/default.aspx",
    "http://academic.openu.ac.il/cs/msc/pages/default.aspx",
    "http://academic.openu.ac.il/teachers/Pages/Prices_and_benefits.aspx",
    "http://academic.openu.ac.il/more-than-degree/Pages/choose_degree_AR.aspx",
    "http://www.openu.ac.il/transfertrack/technion/pages/mechanical_engineering.aspx",
    "http://www.openu.ac.il/alternativeassessment/Pages/arabic_20.aspx",
    "http://www.openu.ac.il/deanacademicstudies/teachandlearn/learning_skiils/pages/exam.aspx",
    "http://www.openu.ac.il/deanacademicstudies/teachandlearn/learning_skiils/pages/arabic_ishi_english.aspx",
    "http://academic.openu.ac.il/cs/computer/pages/default.aspx",
]

# Output directory
output_dir = "FineTuning/files/"

# Set up Google API credentials and services
SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/documents']
GOOGLE_SERVICE_ACCOUNT_FILE = 'credentials.json'

credentials = service_account.Credentials.from_service_account_file(GOOGLE_SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)
docs_service = build('docs', 'v1', credentials=credentials)

# Folder ID in Google Drive where files will be stored (create manually in Google Drive and get the ID)
google_file_id = '1IaTBoXlfxlHFH9LHW8_zFX11GV1Fns5y32sOsiQLrsc'

# Initialize Chrome for Selenium
chrome_service = Service(executable_path='C:/Windows/System32/chromedriver.exe')  # Update the path to your chromedriver
driver = webdriver.Chrome(service=chrome_service)

encoding = tiktoken.get_encoding("cl100k_base")

jsonl_file = "training_data.jsonl"
training_file_id = "file-W8Sepo8jaxn6HdULHJp2PR"

##############################################################################
# Function to clear Google Doc content
def clear_google_doc(doc_id):
    # First, fetch the document to get its end index
    document = docs_service.documents().get(documentId=doc_id).execute()

    # Extract the actual end index from the last element in the content.
    end_index = document["body"]["content"][-1]["endIndex"]
    if end_index is None or end_index <= 1:
        print("Document is already empty or no valid content to clear.")
        return

    # Subtract 1 to avoid including the newline character at the end of the segment
    end_index_adjusted = end_index - 1

    # Create the delete request with the adjusted end index
    requests = [
        {
            "deleteContentRange": {
                "range": {
                    "startIndex": 1,
                    "endIndex": end_index_adjusted
                }
            }
        }
    ]
    docs_service.documents().batchUpdate(documentId=doc_id, body={'requests': requests}).execute()

##############################################################################
def download_google_doc_as_markdown(file_id):
    """
    Given a Google Doc file ID, this function downloads the file's content
    as Markdown. It first attempts to export using MIME type 'text/markdown'
    (the equivalent of choosing "Download as Markdown" in the UI). If that
    fails, it falls back to exporting as HTML and converts it to Markdown.
    """
    # Try to export the Google Doc directly as Markdown.
    try:
        # Attempt to use the Markdown MIME type.
        exported = drive_service.files().export(
            fileId=file_id, mimeType='text/markdown'
        ).execute()
        # The API may return bytes, so decode if necessary.
        if isinstance(exported, bytes):
            markdown = exported.decode('utf-8')
        else:
            markdown = exported
        print("Exported using text/markdown MIME type.")
    except Exception as e:
        print("Markdown export not available; falling back to HTML conversion. Error:", e)
        # Fall back: export as HTML.
        exported = drive_service.files().export(
            fileId=file_id, mimeType='text/html'
        ).execute()
        if isinstance(exported, bytes):
            html_content = exported.decode('utf-8')
        else:
            html_content = exported
        # Convert HTML to Markdown.
        converter = html2text.HTML2Text()
        converter.ignore_links = False  # change as needed
        markdown = converter.handle(html_content)
    
    return markdown

##############################################################################
# Function to export Google Doc to markdown
def export_google_doc_to_md(doc_id, file_number):
    import shutil
    # Download the Google Doc as "markdown.md" in the user's Downloads folder
    markdown = download_google_doc_as_markdown(doc_id)
    destination_file = os.path.join(output_dir, f"{file_number}.md")
    with open(destination_file, 'w', encoding='utf-8') as md_file:
        md_file.write(markdown)
    print(f"Markdown saved to {destination_file}")

##############################################################################
# Main workflow
def save_files_from_urls():
    for idx, url in enumerate(urls, 1):
        # Step 1: Download HTML content
        response = requests.get(url)
    html_content = response.text
    
    # Step 2: Save HTML content
    html_file_path = f'{output_dir}{idx}.html'
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML saved as {html_file_path}")
    
    # Step 3: Clear Google Doc content for next URL
    clear_google_doc(google_file_id)

    # Step 4: Open URL in Chrome for manual work
    driver.get(url)
    print(f"Please copy the content from {url} into the Google Doc.")
    input("Press Enter after pasting the content...")

    # Step 5: Export Google Doc to markdown
    export_google_doc_to_md(google_file_id, idx)
    
    driver.quit()
    print("All URLs processed.")


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
def create_fine_tuning_file():
    print("Creating fine-tuning file...")

    output_jsonl_path = os.path.join(output_dir, "training_data.jsonl")
    
    # List to contain multiple training examples; here we'll build one per file pair.
    # If you want to add more examples later, just adjust the range accordingly.
    training_examples = []
    escaped_system_instructions = escape_content(system_instructions)
    
    for idx in range(1, 37):
        html_file_path = os.path.join(output_dir, f"{idx}.html")
        markdown_file_path = os.path.join(output_dir, f"{idx}.md")

        # Read the HTML and Markdown files
        if not os.path.exists(html_file_path):
            print(f"Skipping {html_file_path}, file does not exist.")
            continue
        print(f"Processing file {idx} of 36")
        with open(html_file_path, 'r', encoding='utf-8') as html_file:
            full_html_content = html_file.read()
            html_content = extract_html_body(full_html_content)
            if not html_content:
                print(f"{html_file_path}: extract_html_body() failed.")
                continue
            print(f"{html_file_path}: HTML size reduced from {len(full_html_content)} to {len(html_content)}")

        with open(markdown_file_path, 'r', encoding='utf-8') as md_file:
            markdown_content = md_file.read()

        # Escape HTML and Markdown content
        escaped_html = escape_content(html_content)
        escaped_markdown = escape_content(markdown_content)

        # Create a training example where each example is a separate JSON object.
        training_example = {
            "messages": [
                {"role": "system", "content": escaped_system_instructions},
                {"role": "user", "content": escaped_html},
                {"role": "assistant", "content": escaped_markdown}
            ]
        }
        training_examples.append(training_example)
    
    # Write each training example as its own line in the JSONL file.
    with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        for example in training_examples:
            jsonl_file.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"JSONL file created at: {output_jsonl_path}")

##############################################################################
def validate_data():
    token_limit = 64*1024 # for gpt-4o
    data_path = os.path.join(output_dir, jsonl_file)

    # Load the dataset
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    # Initial dataset stats
    print("Num examples:", len(dataset))

    # Format error checks
    format_errors = defaultdict(int)

    lines_found = 0
    for ex in dataset:
        lines_found += 1
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
            
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
            
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1
            
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
                
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1
        
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")

    # Warnings and tokens counts
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))
        
    print("token sizes:")
    for idx in range(1, lines_found+1):
        is_oversize = convo_lens[idx-1] > token_limit
        print(f"File {idx} has {convo_lens[idx-1]} tokens, {'oversize' if is_oversize else ''}")
    max_tokens = max(convo_lens)
    print(f"Max tokens: {max_tokens}")

    print("Num examples missing system message:", n_missing_system)
    print("Num examples missing user message:", n_missing_user)
    print_distribution(n_messages, "num_messages_per_example")
    print_distribution(convo_lens, "num_total_tokens_per_example")
    print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
    n_too_long = sum(l > token_limit for l in convo_lens)
    print(f"\n{n_too_long} examples may be over the {token_limit} token limit, they will be truncated during fine-tuning")

    # Pricing and default n_epochs estimate
    MAX_TOKENS_PER_EXAMPLE = token_limit

    TARGET_EPOCHS = 3
    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    MIN_DEFAULT_EPOCHS = 1
    MAX_DEFAULT_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
    print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")


##############################################################################
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

##############################################################################
def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

##############################################################################
def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

##############################################################################
def upload_training_file():
    client = OpenAI()
    file_path = os.path.join(output_dir, jsonl_file)
    
    # Open the file in binary read mode.
    with open(file_path, "rb") as training_file:
        # Upload the file to OpenAI.
        upload_response = client.files.create(file=training_file, purpose="fine-tune")
    

    # Extract and return the file ID.
    training_file_id = upload_response.id
    print(f"Uploaded training file ID: {training_file_id}")
    return training_file_id

##############################################################################
def run_fine_tune_job(training_file_id):
    start_time = datetime.now()

    job = openai_client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model="gpt-4o-mini-2024-07-18",
        # method={
        #     "type": "dpo",
        #     "dpo": {
        #         "hyperparameters": {"beta": 0.1},
        #     },
        # },
    )    

    end_time = datetime.now()
    print(f"Total time: {end_time - start_time}")

    print(job)

##############################################################################
def test():
    url = "https://www.openu.ac.il/counseling/pages/Counseling_to_Build_a_Cur.aspx"
    response = requests.get(url)
    full_html_content = response.text
    html_content = extract_html_body(full_html_content)
    if not html_content:
        print(f"{url}: extract_html_body() failed.")
        return
    print(f"{url}: HTML size reduced from {len(full_html_content)} to {len(html_content)}")
    escaped_html = escape_content(html_content)

    system = escape_content(system_instructions)

    try:
        completion = openai_client.chat.completions.create(
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
    print(md)
    md_file = os.path.join(output_dir, "test.md")
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md)

##############################################################################
if __name__ == "__main__":
#    save_files_from_urls()
#    create_fine_tuning_file()
#    validate_data()
#    training_file_id = upload_training_file()
#    run_fine_tune_job(training_file_id)
    test()
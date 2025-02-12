import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import webbrowser  # New import

# List of URLs
urls = [
    "https://academic.openu.ac.il/cs/bsc_excellence/pages/programs.aspx",
    "https://academic.openu.ac.il/cs/bsc_excellence/pages/interested.aspx",
    # Add more URLs here
]

# Output directory
output_dir = "FineTuning/files/"

# Set up Google API credentials and services
SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/documents']
GOOGLE_SERVICE_ACCOUNT_FILE = 'credentials.json'

credentials = service_account.Credentials.from_service_account_file(GOOGLE_SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)
docs_service = build('docs', 'v1', credentials=credentials)

# Use manually created Google Doc ID
google_file_id = '1IaTBoXlfxlHFH9LHW8_zFX11GV1Fns5y32sOsiQLrsc'

# Initialize Chrome for Selenium
chrome_service = Service(executable_path='C:/Windows/System32/chromedriver.exe')  # Update the path to your chromedriver
driver = webdriver.Chrome(service=chrome_service)

# Function to clear Google Doc content
def clear_google_doc(doc_id):
    requests_body = [
        {"deleteContentRange": {"range": {"startIndex": 1, "endIndex": 1_000_000}}}
    ]
    docs_service.documents().batchUpdate(documentId=doc_id, body={'requests': requests_body}).execute()

# Function to export Google Doc to markdown
def export_google_doc_to_md(doc_id, file_number):
    file = drive_service.files().get(fileId=doc_id, fields='exportLinks').execute()
    export_link = file['exportLinks']['text/plain']
    
    response = requests.get(export_link, headers={'Authorization': f'Bearer {credentials.token}'})
    md_file_path = f'{output_dir}{file_number}.md'
    
    with open(md_file_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
    
    print(f"Markdown file saved as {md_file_path}")

# Main workflow: use the existing Google Doc with the provided ID
doc_id = google_file_id

# Optionally, open the Google Doc in the browser for editing.
google_doc_url = f"https://docs.google.com/document/d/{doc_id}/edit"
print(f"Please open your Google Doc here for editing: {google_doc_url}")
webbrowser.open(google_doc_url)

for idx, url in enumerate(urls, 1):
    # Step 1: Download HTML content
    response = requests.get(url)
    html_content = response.text
    
    # Step 2: Save HTML content
    html_file_path = f'{output_dir}{idx}.html'
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML saved as {html_file_path}")
    
    # Step 3: Open URL with Selenium for manual work
    driver.get(url)
    print(f"Please copy the content from {url} into the Google Doc manually.")
    input("Press Enter after pasting the content...")

    # Step 4: Export Google Doc to markdown
    export_google_doc_to_md(doc_id, idx)
    
    # Step 5: Clear Google Doc content for next URL
    clear_google_doc(doc_id)

driver.quit()
print("All URLs processed.") 
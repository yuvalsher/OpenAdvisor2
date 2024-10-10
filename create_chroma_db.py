# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil
import json
from uuid import uuid4

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
MY_OPENAI_KEY = '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'

# openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_key = MY_OPENAI_KEY

CHROMA_PATH = "OpenAdvisor2/kb/chroma"
KB_PATH = "OpenAdvisor2/kb/json_source"

ord = 1   

##############################################################################
def save_to_chroma(chunks: list[Document], faculty: str):
    # Clear out the database first.
    chrome_db = f"{CHROMA_PATH}_{faculty}"
    if os.path.exists(chrome_db):
        shutil.rmtree(chrome_db)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=chrome_db
    )

    #db.persist()
    print(f"Saved {len(chunks)} chunks to {chrome_db}.")

##############################################################################
def load_json_file(filename, type, documents):
    global ord
    # Load crawled_data.json
    filename = os.path.join(KB_PATH, filename)

    # if the file does not exist - exists the function
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return
    
    with open(filename, "r", encoding='utf-8') as f:
        data = json.load(f)

    local_counter = 0
    course_number = 0
    for item in data:
        md={"type": type}
        if (type != "Courses"):
            content = item["summary"]
            md["source"] = item["url"]
        else:
            content = prepare_course_content(item)
            md["source"] = item["course_url"]
            md["course_number"] = item["course_id"]

        doc = Document(
            page_content = content,
            metadata=md,
            id=ord
        )
        ord += 1
        local_counter += 1
        documents.append(doc)
    print(f"Loaded {local_counter} documents of type {type}.")

##############################################################################
def add_list(name, list):
    if list:
        content =  name
        for item in list:
            content += item + ", "
        content += "\n"
        return content
    else:
        return ""
    
##############################################################################
def prepare_course_content(item):
    content  =  "שם הקורס: " + item["course_name"] + "\n"
    content +=  "מספר הקורס: " + item["course_id"] + "\n"
    content +=  "נקודות זכות: " + item["credits"] + "\n"
    content +=  "שיוך ראשי: " + item["primary_classification"] + "\n"
    content += add_list("שיוך משני: ", item["secondary_classification"])
    content += add_list("קדם נדרש: ", item["required_dependencies"])
    content += add_list("קדם מומלץ: ", item["recommended_dependencies"])
    content += add_list("פרטים: ", item["text"])
    return content

##############################################################################
def create_kb_db(faculty_list):
    for faculty in faculty_list:
        print(f"\nCreating Chroma DB for {faculty}...")
        documents = []
        load_json_file(f"crawled_data_{faculty}.json", "web-text", documents)
        load_json_file(f"youtube_transcripts_{faculty}.json", "YouTube", documents)
        if (faculty != "OUI"):
            load_json_file(f"{faculty}_courses.json", "Courses", documents)

        #uuids = [str(uuid4()) for _ in range(len(documents))]
        save_to_chroma(documents, faculty)
        print(f"Created Chroma DB for {faculty}.")

##############################################################################
if __name__ == "__main__":
    create_kb_db(["CS", "OUI"])

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
from config import all_config

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
MY_OPENAI_KEY = all_config["General"]["OPENAI_API_KEY"]

# openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_key = MY_OPENAI_KEY
ord = 1  
##############################################################################

##############################################################################
def save_to_chroma(chunks: list[Document], faculty: str):
    # Get the general configuration
    general_config = all_config["General"]

    # Use the Chroma_Path from the config
    chrome_db = f"{general_config['Chroma_Path']}_{faculty}"
    if os.path.exists(chrome_db):
        shutil.rmtree(chrome_db)

    embeddings = OpenAIEmbeddings(
        model=general_config["embeddings"],
        openai_api_key=MY_OPENAI_KEY ###general_config["OPENAI_API_KEY"]
    )

    # Create a new DB from the documents.
    batch_size = 300  # Adjust based on document size and token count
    for i in range(0, len(chunks), batch_size):
        print(f"Processing batch from {i} to {i + batch_size}")
        batch = chunks[i:i + batch_size]
        Chroma.from_documents(batch, embeddings, persist_directory=chrome_db)

#    db = Chroma.from_documents(
#        chunks, embeddings, persist_directory=chrome_db
#    )

    print(f"Saved {len(chunks)} chunks to {chrome_db}.")

##############################################################################
def check_classification(c, classifications):
    if not c:
        return False
    for c1 in c:
        if not c1 or len(c1) < 2:
            continue
        if c1[1] in classifications:
            return True
        
    return False

##############################################################################
def load_json_file(filename, type, documents, classifications=None):
    global ord
    # Get the general configuration
    general_config = all_config["General"]

    # Use the DB_Path from the config
    full_path = os.path.join(general_config["DB_Path"], filename)

    # if the file does not exist - exits the function
    if not os.path.exists(full_path):
        print(f"File {full_path} does not exist.")
        return
    
    with open(full_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    local_counter = 0
    course_number = 0
    for item in data:
        md={"type": type}
        if (type != "Courses"):
            content = item["summary"]
            md["source"] = item["url"]
        else:
            if classifications and not check_classification(item["classification"], classifications):
                continue

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
    # content +=  "נקודות זכות: " + item["credits"] + "\n"
    # classification = item["classification"]
    # for c in classification:
    #     content += add_list("שיוך: ", c)
    # content += add_list("קדם נדרש: ", item["required_dependencies"])
    # content += add_list("קדם מומלץ: ", item["recommended_dependencies"])
    content += add_list("פרטים: ", item["text"])
    return content

##############################################################################
def create_kb_db(faculty_list):
    for faculty in faculty_list:
        print(f"\nCreating Chroma DB for {faculty}...")
        documents = []
        load_json_file(f"crawled_data_{faculty}.json", "web-text", documents)
        #load_json_file(f"youtube_transcripts_{faculty}.json", "YouTube", documents)
        if (faculty != "OUI"):
            classifications = None
            if (faculty == "CS"):
                classifications = ["מדעי המחשב", "מדעי המחשב תואר שני", "הנדסת תוכנה", "למידת מכונה וניתוח נתוני עתק"]
            #load_json_file(f"{faculty}_courses.json", "Courses", documents, classification)
            load_json_file(f"all_courses.json", "Courses", documents, classifications)

        save_to_chroma(documents, faculty)
        print(f"Created Chroma DB for {faculty}.")

##############################################################################
def create_courses_db():
    print(f"\nCreating Chroma DB for all courses...")
    documents = []
    load_json_file("all_courses.json", "Courses", documents)

    save_to_chroma(documents, "Courses")
    print(f"Created Chroma DB for all courses.")

##############################################################################
def course_stats():
    # Get the general configuration
    general_config = all_config["General"]
    # Use the DB_Path from the config
    full_path = os.path.join(general_config["DB_Path"], "all_courses.json")

    # if the file does not exist - exits the function
    if not os.path.exists(full_path):
        print(f"File {full_path} does not exist.")
        return
    
    with open(full_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    # Find the min and max course numbers
    min_course = '99999'
    max_course = '0'
    deps = {}
    total_courses = len(data)
    for item in data:
        classifications = item['classification']
        for c in classifications:
            c0 = c[0]
            if not c0:
                print(f"Empty classification for {item['course_id']}:{item['course_name'][::-1]}")
                continue

            if c0 not in deps:
                deps[c0] = { 'courses': [], 'faculties': {} }
            id = item['course_id']
            # check if the course number is a number
            if not id.isdigit():
                print(f"Course {id} is not a number: {item['course_name'][::-1]}")
                continue
            else:
                deps[c0]['courses'].append(id)

            if id < min_course:
                min_course = id
            if id > max_course:
                max_course = id 

            if len(c) > 1:
                c1 = c[1]
                if c1 not in deps[c0]['faculties']:
                    deps[c0]['faculties'][c1] = { 'courses': [] }
            deps[c0]['faculties'][c1]['courses'].append(id)

    print(f"Total courses: {total_courses} from {min_course} to {max_course}")
    for d in deps:
        courses = deps[d]['courses']
        min_courses = min(courses)
        max_courses = max(courses)
        print(f"{d[::-1]}: {len(courses)} courses from {min_courses} to {max_courses}")
        #print(f"{d}: {len(courses)} courses from {min_courses} to {max_courses}")
        for f in deps[d]['faculties']:
            f_courses = deps[d]['faculties'][f]['courses']
            min_f_courses = min(f_courses)
            max_f_courses = max(f_courses)
            print(f"    {f[::-1]}: {len(f_courses)} courses from {min_f_courses} to {max_f_courses}")
            #print(f"    {f}: {len(f_courses)} courses from {min_f_courses} to {max_f_courses}")


##############################################################################
if __name__ == "__main__":
    #create_kb_db(["CS", "OUI"])
    create_courses_db()
    #course_stats()

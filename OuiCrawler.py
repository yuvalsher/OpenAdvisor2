import requests
from bs4 import BeautifulSoup, Comment, NavigableString
import json
from urllib.parse import urljoin, urlparse
import re
import os
import json
from openai import OpenAI
from datetime import datetime

from config import all_config, all_crawl_config
from YouTubeTools import YouTubeTools

crawl_config = {}

"""
faq_pages = [
    'https://academic.openu.ac.il/cs/computer/pages/faq.aspx',
    'https://academic.openu.ac.il/education_psychology/leadership/pages/questions.aspx',
    'https://academic.openu.ac.il/education_psychology/laboratory/pages/questions.aspx',
    'https://academic.openu.ac.il/history/history/pages/faq.aspx',
    'https://academic.openu.ac.il/localgovernmentschool/pages/common.aspx',
    'https://www.openu.ac.il/transfertrack/pages/faq.aspx',
    'https://www.openu.ac.il/dean-students/ld/pages/faq.aspx',
    'https://academic.openu.ac.il/cs/bsc_excellence/pages/faq.aspx'
]
"""

# Set to store visited URLs to avoid revisiting
visited_urls = set()

# List to store page data
pages_data = []
pages_dict = {}

youtube_links = []
msg_log = []
pdf_files = []

is_testing = False
context = ""

##############################################################################
def is_allowed_domain(url):
    global crawl_config

    if is_testing:
        return True
    
    # Parse the URL to get the domain and path
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path

    # check if web page starts with any of the disallowed_pages
    for disallowed_page in crawl_config['disallowed_pages']:
        if url.startswith(disallowed_page):
            return False    

    # Check if the full domain and path are in the disallowed domains
    for disallowed_domain in crawl_config['disallowed_domains']:
        disallowed_parsed = urlparse(f"https://{disallowed_domain}")
        disallowed_domain_netloc = disallowed_parsed.netloc
        disallowed_domain_path = disallowed_parsed.path

        # Check if the domain matches and the path starts with the allowed path
        if domain == disallowed_domain_netloc and path.startswith(disallowed_domain_path):
            return False

    # Check if the full domain and path are in the allowed domains
    for allowed_domain in crawl_config['allowed_domains']:
        allowed_parsed = urlparse(f"https://{allowed_domain}")
        allowed_domain_netloc = allowed_parsed.netloc
        allowed_domain_path = allowed_parsed.path

        # Check if the domain matches and the path starts with the allowed path
        if domain == allowed_domain_netloc and path.startswith(allowed_domain_path):
            return True

    return False

##############################################################################
chunk_styles = [
    {
        'name': 'CS',
        'heading': 'panel panel-default',
        'title': 'panel-heading',
        'content': 'panel-body'
    },
    {
        'name': 'Dean',
        'heading': 'course-programm-item',
        'title': 'course-programm-title',
        'content': 'course-programm-content'
    }
]

##############################################################################
def break_into_chunks(element):
    chunk_data = []

    for style in chunk_styles:
        chunks = element.find_all('div', {'class': style['heading']})
        if chunks:
            for chunk in chunks:
                title_elem = chunk.find('div', {'class': style['title']})
                title = title_elem.text if title_elem else ''
                content_elem = chunk.find('div', {'class': style['content']})
                content = content_elem.text if content_elem else ''
                chunk_content = title + '\n' + content
                # remove multiple occurances of '\n' characters in the text_content, and replace with a single '\n'.
                chunk_content = re.sub(r'\n+', '\n', chunk_content)
                chunk_data.append(chunk_content)
            return chunk_data

    return None

##############################################################################
def contains_chunks(element):
    container = element.find('div', {'class': 'RowContainer RowContainerNumber_2'})
    if not container:
        container =  element.find('div', {'class': 'ZoneContainer'})
    return container

##############################################################################
# Extract the text content of the web page. Use only stuff after the end of the header, marked with the '<!--END HEADER-->' element, and before the footer, marked with the "<!--FOOTER-->" element.
def extract_text_content(soup, url, todo_links):
    # Find all comments
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    # Locate the 'END HEADER' and 'FOOTER' comments
    end_header_comment = None
    footer_comment = None

    for comment in comments:
        if comment == "END HEADER":
            end_header_comment = comment
        elif comment == "FOOTER":
            footer_comment = comment

    # Ensure both comments are found
    text_content = ""
    chunks = []
    if end_header_comment and footer_comment:
        # Get the next sibling of the 'END HEADER' comment
        current_element = end_header_comment.next_sibling

        # Iterate until the 'FOOTER' comment is reached
        while current_element and current_element != footer_comment:
            # Process the current element
            if isinstance(current_element, Comment):
                current_element = current_element.next_sibling
                continue

            found_chunks = False
            # check if current_element is a BeautifulSoup element
            if not isinstance(current_element, NavigableString):
                container = contains_chunks(current_element)
                # find if the current element contains the element "div class="RowContainer RowContainerNumber_2"
                if container:
                    elem_chunks = break_into_chunks(container)
                    if elem_chunks:
                        chunks.extend(elem_chunks)
                        found_chunks = True
                    else:
                        msg = "Found a RowContainer, but no chunks in it: " + url + "\n"
                        #msg_log.append(msg)

            if not found_chunks:
                text_content += current_element.text

            if not is_testing:
                check_for_links(current_element, url, todo_links)

            current_element = current_element.next_sibling

    # remove multiple occurances of '\n' characters in the text_content, and replace with a single '\n'.
    text_content = re.sub(r'\n+', '\n', text_content)
    if (len(text_content) > 2):
        chunks.append(text_content)
    return chunks


##############################################################################
def extract_youtube_transcript(soup):
    transcript = ""
    transcript_tag = soup.find('div', {'id': 'transcript'})
    if transcript_tag:
        transcript = transcript_tag.get_text(separator=' ', strip=True)
    return transcript

##############################################################################
def add_chunk(url, title, chunk, type):
    summary = summarize_text(title + "\n" + chunk)
    pages_data.append({
        'url': url,
        'title': title,
        'context': context,
        'type': type,
        'text_content': chunk,
        'summary': summary
    })

##############################################################################
def crawl_page(url, todo_links):
    global pages_data
    global pages_dict

    # Check if the URL points to a PDF file
    if url.lower().endswith('.pdf'):
        pdf_files.append(url)
        msg = f"Skipping PDF file: {url}"
        print(msg)
        return

    # Check if the URL points to a PDF file
    if url.lower().endswith('.docx'):
        pdf_files.append(url)
        msg = f"Skipping DOCX file: {url}"
        print(msg)
        return

    # If URL is not in allowed domain or already visited, skip
    if url in visited_urls or not is_allowed_domain(url):
        return
    
    # if visited_urls contains a url that differs from the parameter url only by case, skip
    for visited_url in visited_urls:
        if visited_url.lower() == url.lower():
            return

    # Mark the URL as visited
    visited_urls.add(url)

    try:
        # Make a request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad requests
    except requests.RequestException as e:
        #print(f"Failed to access {url}: {e}")
        return

    # Parse the page content with BeautifulSoup
    try:    
        soup = BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        msg = f"Failed to parse {url}: {e}"
        print(msg)
        msg_log.append(msg)
        return

    # Extract the page title
    title = soup.title.string if soup.title else ''

    text_content = extract_text_content(soup, url, todo_links)

    for chunk in text_content:
        if chunk in pages_dict:
            continue 
        else:
            pages_dict[chunk] = url

    if title:
        print(f"Added page: {title[::-1]} @ {url}\n")
    else:
        msg = f"Page with empty title: {url}\n"
        print(msg)
        msg_log.append(msg)

    # Save the page data
    if text_content:
        url_tip = ""
        is_single = True
        if (len(text_content) > 1):
            is_single = False
        i = 1    
        for chunk in text_content:
            if not is_single:
                url_tip = f"?{i}"
                i+=1
            add_chunk(url+url_tip, title, chunk, "page")

    # Find all links to YouTube videos that look like this:
    # <link rel="canonical" href="https://www.youtube.com/watch?v=mPBFHQ4X31s">
    iframes = soup.find_all('iframe', {'src': re.compile(r'https://www.youtube.com/embed/.*?wmode=transparent')})

    for link in iframes:
        link_url = link['src']
        # extract the video id from the src string in the format: "https://www.youtube.com/embed/kUCM6uEwP-Q?wmode=transparent"
        video_id = link_url.split('/')[-1].split('?')[0]
        video_link = f"https://www.youtube.com/watch?v={video_id}"
        if video_link not in youtube_links:
            youtube_links.append(video_link)


##############################################################################
def check_for_links(element, url, todo_links):
    if isinstance(element, NavigableString):
        return

    # Find all links on the current page and crawl them
    for link in element.find_all('a', href=True):
        next_url = urljoin(url, link['href'])
        # Normalize and remove URL fragments (e.g., #section)
        next_url = urlparse(next_url)._replace(fragment='').geturl()
        #  Add link to the ToDo list
        if next_url not in todo_links and is_allowed_domain(next_url):
            todo_links.append(next_url)

##############################################################################
# Function for reading the output jaon file and providing statistics on it
# Print the following output:
# 1. How many pages were collected in total?
# 2. How many pages were collected for each of the allowed_domains
def do_stats():

    # Count the total number of pages collected
    total_pages = len(pages_data)

    # Count the number of pages collected for each allowed domain
    domain_counts = {}
    for domain in crawl_config['allowed_domains']:
        domain_counts[domain] = sum(1 for page in pages_data if domain in page['url'])

    # Print the statistics
    print(f"Total pages collected: {total_pages}")
    for domain, count in domain_counts.items():
        print(f"Pages collected for {domain}: {count}")

##############################################################################
def do_all_stats():
    global pages_data
    # Count the total number of pages collected
    total_pages = len(pages_data)
    if (total_pages == 0):
        # Read the JSON file
        db_path = all_config['General']['DB_Path']
        filename = os.path.join(db_path, "crawled_data_All.json")
        with open(filename, 'r', encoding='utf-8') as file:
            pages_data = json.load(file)
            total_pages = len(pages_data)
        
    domains = {}
    ac_domains = {}
    dom1 = ""
    dom2 = ""
    ac_pages = 0
    main_pages = 0
    not_counted = []
    youtube_links = []
    EMPTY_DOM = '<>'
    for page in pages_data:
        url = page['url']
        parsed_url = urlparse(url)
        dom = parsed_url.netloc
        path = parsed_url.path
        dom1 = path.split('/')[1] if len(path.split('/')) > 1 else EMPTY_DOM
        if (dom1 == ''):
            dom1 = EMPTY_DOM
        dom2 = path.split('/')[2] if len(path.split('/')) > 2 else EMPTY_DOM
        if (dom2 == '' or dom2.endswith('.aspx') or dom2.endswith('.html') or dom2.endswith('.htm')):
            dom2 = EMPTY_DOM

        if dom == 'www.openu.ac.il':
            main_pages += 1
            if not dom1 in domains:
                domains[dom1] = {}
            if not dom2 in domains[dom1]:
                domains[dom1][dom2] = 0
            domains[dom1][dom2] += 1
        elif dom == 'academic.openu.ac.il':
            ac_pages += 1
            if not dom1 in ac_domains:
                ac_domains[dom1] = {}
            if not dom2 in ac_domains[dom1]:
                ac_domains[dom1][dom2] = 0
            ac_domains[dom1][dom2] += 1
        elif dom == 'www.youtube.com':
            youtube_links.append(url)
        else:
            not_counted.append(url)

    print(f"Total pages collected: {total_pages}\n")
    print(f"Main pages collected: {main_pages}")
    for dom1 in domains:
        print(f"Pages collected for {dom1}:")
        dom1_total = 0
        for dom2 in domains[dom1]:
            if dom2 != EMPTY_DOM:
                print(f"    {dom2}: {domains[dom1][dom2]}")
            dom1_total += domains[dom1][dom2]
        print(f"    Total: {dom1_total}")

    print(f"\n\nAcademic pages collected: {ac_pages}")
    for dom1 in ac_domains:
        print(f"Pages collected for {dom1}:")
        dom1_total = 0
        for dom2 in ac_domains[dom1]:
            if dom2 != EMPTY_DOM:
                print(f"    {dom2}: {ac_domains[dom1][dom2]}")
            dom1_total += ac_domains[dom1][dom2]
        print(f"    Total: {dom1_total}")

    print(f"\n\nYoutube links collected: {len(youtube_links)}:")
    for url in youtube_links:
        print(f"    {url}")

    print(f"\n\nNot counted pages: {len(not_counted)}:")
    for url in not_counted:
        print(f"    {url}")

    print(f"\n\nPDF\DOCX files collected: {len(pdf_files)}")
    for pdf in pdf_files:
        print(f"    {pdf}")

                

##############################################################################
# loop over youtube links, and use the YouTube API to get the video transcript.
# If the transcript is found, it is added to the pages_data list.
def check_videos_for_transcripts(youtube_links):
    msg = "\nFound the following YouTube links:"
    print(msg)
    msg_log.append(msg)
    for vid in youtube_links:
        video_id = YouTubeTools.get_video_id(vid)
        if video_id == 'videoseries':
            continue
        
        full_text = YouTubeTools.fetch_youtuve_transcript(video_id)
        title = YouTubeTools.fetch_youtube_title(vid)

        msg = ""
        if full_text:
            if title is None:
                title = "Unknown Title"
            add_chunk(vid, title, full_text, "video")
            msg = f"{vid} - Transcript found, length: {len(full_text)}, title: {title[::-1]}."
        else:
            msg = f"{vid} - No transcript found."
        
        print(msg)
        msg_log.append(msg)

##############################################################################
def start_crawling(faculties):
    global pages_data, crawl_config, context

    start_time = datetime.now()
    print(f"Starting at: {start_time}")
    msg_log.append(f"Starting at: {start_time}")

    for faculty in faculties:
        msg = f"\n\nStarting crawling for {faculty}...\n"
        print(msg)
        msg_log.append(msg)
        context = faculty
        crawl_config = all_crawl_config[faculty]

        todo_links = []
        # Start crawling from the initial URLs
        for url in crawl_config['start_urls']:
            while True:
                crawl_page(url, todo_links)

                if len(todo_links) == 0:
                    break
                else:
                    url = todo_links.pop(0)

        check_videos_for_transcripts(youtube_links)

        # Save the collected data to a JSON file
        db_path = all_config['General']['DB_Path']
        filename = os.path.join(db_path, "crawled_data_" + faculty + ".json")
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(pages_data, json_file, ensure_ascii=False, indent=4)

        if (faculty == "All"):
            do_all_stats()
        else:
            do_stats()

        end_time = datetime.now()
        print(f"Ending at: {end_time}, total time: {end_time - start_time}")
        msg_log.append(f"Ending at: {end_time}, total time: {end_time - start_time}")

        print("\nLog Messages:")
        for msg in msg_log:
            print(msg)


    # Add the number of PDF files, and then the list of PDF files to msg_log
    msg_log.append(f"\nPDF files collected: {len(pdf_files)}\n")
    for pdf in pdf_files:
        msg_log.append(f"    {pdf}\n")

    # write msg_log to log file
    log_filename = os.path.join(db_path, "crawled_data_" + faculty + "_log.txt")
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        for msg in msg_log:
            log_file.write(msg + "\n")

##############################################################################
def test_crawling():
    global pages_data
    global is_testing

    is_testing = True
    #crawl_page('https://academic.openu.ac.il/education_psychology/laboratory/pages/questions.aspx')
    #for page in faq_pages:
    #    crawl_page(page)


    # Save the collected data to a JSON file
    with open('test_crawled_data.json', 'w', encoding='utf-8') as json_file:
        json.dump(pages_data, json_file, ensure_ascii=False, indent=4)

    do_stats()


##############################################################################
def summarize_text(text, model="gpt-4o-mini", max_tokens=500):
    """
    Summarizes the input text using OpenAI's API.
    
    Args:
    - text (str): The text to summarize.
    - model (str): The OpenAI model to use (default is "gpt-4").
    - max_tokens (int): The maximum number of tokens for the summary (default is 300).
    
    Returns:
    - str: The summary of the input text.
    """
    
    # Set the OpenAI API key
    openai_epi_key = all_config["General"]["OPENAI_API_KEY"]
    client = OpenAI(api_key=openai_epi_key)
    
    try:
        # Call the OpenAI API to summarize the text
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Please create a detailed summary of the following text using the input language:\n\n{text}"}
            ]
        )

        # Extract the summary from the response
        summary = response.choices[0].message.content.strip()
        return summary
    
    except Exception as e:
        return f"An error occurred: {e}"


##############################################################################
def test_videos():
    vid_links = ['https://www.youtube.com/watch?v=rnHixxhRWig', 
                 'https://www.youtube.com/watch?v=o4mgsfGHJy8', 
                 'https://www.youtube.com/watch?v=Y2JB5eKB7-Q', 
                 'https://www.youtube.com/watch?v=fPrq1sHu8ug', 
                 'https://www.youtube.com/watch?v=videoseries']
    
    check_videos_for_transcripts(vid_links)

##############################################################################
if __name__ == "__main__":
#    start_crawling(["CS"])
#    test_videos()
#     start_crawling(["CS", "OUI"])
    #start_crawling(["All"])
    do_all_stats()

import asyncio
from dataclasses import dataclass
from bs4 import BeautifulSoup, Comment, NavigableString
from markdownify import markdownify as md
import json
from urllib.parse import urljoin, urlparse
import re
import os
import json
from openai import AsyncOpenAI, OpenAI
from datetime import datetime, timezone
from typing import Any, Dict, List
import aiohttp
import requests
from supabase import create_client, Client
from playwright.async_api import async_playwright

from config import all_config, all_crawl_config
from YouTubeTools import YouTubeTools

debug_mode = False

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
irregular_pages = []

youtube_links = []
msg_log = []
pdf_files = []
disallowed_pages = 0
other_domains = {}

is_testing = False
context = ""

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

##############################################################################
def is_allowed_domain(url):
    global crawl_config, disallowed_pages, other_domains

    if is_testing:
        return True
    
    url = url.lower()
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
            disallowed_pages += 1
            return False

    # Check if the full domain and path are in the allowed domains
    for allowed_domain in crawl_config['allowed_domains']:
        allowed_parsed = urlparse(f"https://{allowed_domain}")
        allowed_domain_netloc = allowed_parsed.netloc
        allowed_domain_path = allowed_parsed.path

        # Check if the domain matches and the path starts with the allowed path
        if domain == allowed_domain_netloc and path.startswith(allowed_domain_path):
            return True

    if domain not in other_domains:
        other_domains[domain] = 0
    other_domains[domain] += 1
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
def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

##############################################################################
async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk using the same language as the chunk.
    Make sure the title and summary do not include single or double quotes characters.
    Both title and summary should be in the same language as the chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        msg = f"Error getting title and summary: {e}"
        print(msg)
        msg_log.append(msg)

        return {"title": "Error processing title", "summary": "Error processing summary"}

##############################################################################
async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        msg = f"Error getting embedding: {e}"
        print(msg)
        msg_log.append(msg)

        return [0] * 1536  # Return zero vector on error

##############################################################################
async def process_chunk(chunk: str, chunk_number: int, url: str, dataset_name: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": dataset_name,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

##############################################################################
async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        table_name = all_config["General"]["supabase_table_name"]
        result = supabase.table(table_name).insert(data).execute()
        msg = f"Inserted chunk {chunk.chunk_number} for {chunk.url}"
        print(msg)
        msg_log.append(msg)
        return result
    except Exception as e:
        msg = f"Error inserting chunk: {e}"
        print(msg)
        msg_log.append(msg)
        return None

##############################################################################
async def clear_supabase_table(dataset_name: str):
    try:
        table_name = all_config["General"]["supabase_table_name"]
        supabase.table(table_name).delete().eq("metadata->>source", dataset_name).execute()
        print(f"Cleared {dataset_name} rows from {table_name} table")
    except Exception as e:
        msg = f"Error clearing {dataset_name} rows from {table_name} table: {e}"
        print(msg)
        msg_log.append(msg)

##############################################################################
async def process_and_store_document(url: str, markdown: str, dataset_name: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url, dataset_name) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

##############################################################################
def is_skip_page(url: str):
    skip_file_types = ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.tiff', '.ico', '.svg', '.webp',
                       '.xlsx', '.pptx', '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz', '.exe', 
                       '.dll', '.so', '.lib', '.a', '.dylib']
    for skip_file_type in skip_file_types:
        if url.lower().endswith(skip_file_type):
            return True

    # Check if the URL points to a PDF file
    if url.lower().endswith('.pdf'):
        pdf_files.append(url)
        msg = f"Skipping PDF file: {url}"
        print(msg)
        msg_log.append(msg)
        return True

    # Check if the URL points to a DOCX file
    if url.lower().endswith('.docx'):
        pdf_files.append(url)
        msg = f"Skipping DOCX file: {url}"
        print(msg)
        msg_log.append(msg)
        return True

    # If URL is not in allowed domain or already visited, skip
    if url in visited_urls or not is_allowed_domain(url):
        return True

    # Ignore difference between "http" and "https"    
    if url.startswith("https"):
        url = url.replace("https", "http", 1)

    # if visited_urls contains a url that differs from the parameter url only by case, skip
    for visited_url in visited_urls:
        if visited_url.lower() == url.lower():
            return True
        
    return False

##############################################################################
async def get_page_content(url: str) -> str:
    html_content = ""

    # First try to get the page using aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    html_content = await response.text()
            except aiohttp.ClientError as e:
                print(f"Connection error for {url}: {str(e)}")
                return None
    except Exception as e:
        print(f"Error accessing {url}: {str(e)}")
        return None
        
    html_body = extract_html_body(html_content)
    if html_body:
        # That was easy!
        return html_body

    # If we got here, this is a dynamic page and we need to use Playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, timeout=30000)
        page = await browser.new_page()
        try:
            # Navigate and wait for network to be idle
            response = await page.goto(url, wait_until='networkidle')
            if response.status != 200:
                return None
            
            # Wait for content to load (adjust selector as needed)
            await page.wait_for_selector('body')

            # Get the rendered HTML
            html_content = await page.content()
            html_body = extract_html_body(html_content)
            if not html_body:
                irregular_pages.append(url)
                return None

            return html_body
        except Exception as e:
            msg = f"Error crawling {url}: {str(e)}"
            print(msg)
            msg_log.append(msg)
            return None
            
        finally:
            await browser.close()


##############################################################################
async def crawl_page(url: str, todo_links: List[str]) -> bool:
    global pages_data
    global pages_dict
    global irregular_pages

    if is_skip_page(url):
        return False

    # Mark the URL as visited
    visited_urls.add(url)

    html_body = await get_page_content(url)
    if not html_body:
        return False
            
    md_text = md(html_body)
    dataset_name = all_config["General"]["dataset_name_pages"]
    await process_and_store_document(url, md_text, dataset_name)

    # Now use BeautifulSoup to search for links
    try:    
        soup = BeautifulSoup(html_body, 'html.parser')
    except Exception as e:
        msg = f"Failed to parse {url}: {e}"
        print(msg)
        msg_log.append(msg)
        return


    await check_for_links(soup, url, todo_links)
    await extract_youtube_links(soup)

    return True

##############################################################################
async def check_for_links(soup: BeautifulSoup, url: str, todo_links: List[str]):
    # Find all links on the current page
    links = soup.find_all('a', href=True)
    
    for link in links:
        href = link['href']
        if href:
            next_url = urljoin(url, href)
            # Normalize and remove URL fragments (e.g., #section)
            next_url = urlparse(next_url)._replace(fragment='').geturl()
            # Skip YouTube URLs since they are handled separately
            if next_url.startswith('https://www.youtube.com') or next_url.startswith('http://www.youtube.com'):
                # Extract video ID from src URL
                video_id = next_url.split('/')[-1].split('?')[0]
                video_link = f"https://www.youtube.com/watch?v={video_id}"
                if video_link not in youtube_links:
                    youtube_links.append(video_link)
                continue

            # Add link to the ToDo list
            if debug_mode:
                return
            
            if next_url not in todo_links and is_allowed_domain(next_url) and next_url not in visited_urls:
                todo_links.append(next_url)

##############################################################################
async def extract_youtube_links(soup:BeautifulSoup):
    # Find all YouTube video iframes
    iframes = soup.find_all('iframe', {'src': re.compile(r'https://www.youtube.com/embed/.*?wmode=transparent')})
    
    for iframe in iframes:
        # Get the src attribute directly from BeautifulSoup element
        link_url = iframe.get('src')  # Use get() instead of get_attribute()
        if link_url:
            # Extract video ID from src URL
            video_id = link_url.split('/')[-1].split('?')[0]
            video_link = f"https://www.youtube.com/watch?v={video_id}"
            if video_link not in youtube_links:
                youtube_links.append(video_link)

##############################################################################
# loop over youtube links, and use the YouTube API to get the video transcript.
# If the transcript is found, it is added to the pages_data list.
async def check_videos_for_transcripts(youtube_links):
    msg = "\nFound the following YouTube links:"
    print(msg)
    msg_log.append(msg)
    for vid in youtube_links:
        video_id = YouTubeTools.get_video_id(vid)
        if video_id == 'videoseries':
            continue
        
        full_text = await YouTubeTools.fetch_youtuve_transcript(video_id)
        title = await YouTubeTools.fetch_youtube_title(vid)

        msg = ""
        if full_text:
            if title is None:
                title = "Unknown Title"
            dataset_name = all_config["General"]["dataset_name_videos"]
            await process_and_store_document(vid, full_text, dataset_name)
            msg = f"{vid} - Transcript found, length: {len(full_text)}, title: {title[::-1]}."
        else:
            msg = f"{vid} - No transcript found."
        
        print(msg)
        msg_log.append(msg)

##############################################################################
async def crawl_parallel(todo_links: List[str], max_concurrent: int = 10):
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_url(url: str):
        async with semaphore:
            result = await crawl_page(url, todo_links)
            if result:
                print(f"Successfully crawled: {url}")
    
    tasks = []
    while todo_links:
        url = todo_links.pop(0)  # Get and remove the first URL
        task = asyncio.create_task(process_url(url))
        tasks.append(task)
        if len(todo_links) == 0:            
            #await asyncio.sleep(20) # Wait a bit to let other work run
            await asyncio.gather(*tasks)

##############################################################################
async def start_crawling(faculties):
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

        # Start crawling from the initial URLs
        todo_links = crawl_config['start_urls'].copy()
        await crawl_parallel(todo_links)

        await check_videos_for_transcripts(youtube_links)

        # Save the collected data to a JSON file
        db_path = all_config['General']['DB_Path']
        filename = os.path.join(db_path, "crawled_data_" + faculty + ".json")
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(pages_data, json_file, ensure_ascii=False, indent=4)

        do_all_stats()

        end_time = datetime.now()
        print(f"Ending at: {end_time}, total time: {end_time - start_time}")
        msg_log.append(f"Ending at: {end_time}, total time: {end_time - start_time}")

        print("\nLog Messages:")
        for msg in msg_log:
            print(msg)


    # Add the number of PDF files, and then the list of PDF files to msg_log
    msg_log.append(f"\nPDF files collected: {len(pdf_files)}\n")
    for pdf in pdf_files:
        msg_log.append(f"    {pdf}")

    # write msg_log to log file
    log_filename = os.path.join(db_path, "crawled_data_" + faculty + "_log.txt")
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        for msg in msg_log:
            log_file.write(msg + "\n")

    # Write irregular_pages to a file
    with open(os.path.join(db_path, "irregular_pages.txt"), 'w', encoding='utf-8') as irregular_file:
        for page in irregular_pages:
            irregular_file.write(page + "\n")

##############################################################################
def base_url_stats():
    #base_urls = {}
    text_counters = {}
    dups = 0
    for page in pages_data:
        url = page['url']
        if url.startswith('https://www.youtube.com/watch?v='):
            continue

        text = page['text_content']
        if text in text_counters:
            dups += 1
        else:
            text_counters[text] = []
        text_counters[text].append(url)
        
    for text in text_counters:
        if len(text_counters[text]) > 1:
            print(f"Text found in {len(text_counters[text])} pages:")
            for url in text_counters[text]:
                print(f"    {url}")

    print(f"Total duplicates found: {dups}, out of total pages: {len(pages_data)}")

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
        
    base_url_stats()
    return
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
    print(f"Main pages collected (under 'www.openu.ac.il'): {main_pages}")
    for dom1 in domains:
        print(f"Pages collected for {dom1}:")
        dom1_total = 0
        for dom2 in domains[dom1]:
            if dom2 != EMPTY_DOM:
                print(f"    {dom2}: {domains[dom1][dom2]}")
            dom1_total += domains[dom1][dom2]
        print(f"    Total: {dom1_total}")

    print(f"\n\nAcademic pages collected (under 'academic.openu.ac.il'): {ac_pages}")
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

    print(f"\n\nPDF and DOCX files collected: {len(pdf_files)}")
    for pdf in pdf_files:
        print(f"    {pdf}")

    print(f"\n\nDisallowed pages: {disallowed_pages}")
    print(f"\nOther domains (not collected):")
    for domain in other_domains:
        print(f"    {domain}: {other_domains[domain]}")


##############################################################################
async def main():
    youtube_links = [
        'https://www.youtube.com/watch?v=DYlZf2LH_qI',
        'https://www.youtube.com/watch?v=Qk92tbegpm0',
        'https://www.youtube.com/watch?v=yNqQxY_TyqQ',
        'https://www.youtube.com/watch?v=C4cACG_NEiA',
        'https://www.youtube.com/watch?v=C5Lm6Pkenw0',
        'https://www.youtube.com/watch?v=yphJeZjjiy4',
        'https://www.youtube.com/watch?v=CbNeE5QCwgk',
        'https://www.youtube.com/watch?v=prUy_34PrtE',
        'https://www.youtube.com/watch?v=vVnDOlZMt4Q',
        'https://www.youtube.com/watch?v=AyOiyzi5n7w',
        'https://www.youtube.com/watch?v=IUfgiUF6Vmw',
        'https://www.youtube.com/watch?v=0y5l5Ku1Q1I',
        'https://www.youtube.com/watch?v=OuiSVhz4Egs',
        'https://www.youtube.com/watch?v=KcooupFjuWo',
        'https://www.youtube.com/watch?v=zfp4qGcmyyY',
        'https://www.youtube.com/watch?v=AryJWdj67mc',
        'https://www.youtube.com/watch?v=drLvR5sHjVw',
        'https://www.youtube.com/watch?v=xHZnhk55qQM',
        'https://www.youtube.com/watch?v=ZevTs7tcgD4',
        'https://www.youtube.com/watch?v=U1La4iXW1nA',
        'https://www.youtube.com/watch?v=1knGhpDAke4',
        'https://www.youtube.com/watch?v=mPBFHQ4X31s',
        'https://www.youtube.com/watch?v=opjeVcPEtcw',
        'https://www.youtube.com/watch?v=kUCM6uEwP-Q',
        'https://www.youtube.com/watch?v=Q5cTZOzKy-A',
        'https://www.youtube.com/watch?v=QBUockBMcEg',
        'https://www.youtube.com/watch?v=lHH27uc66e4',
        'https://www.youtube.com/watch?v=rfC5FRc_RZQ',
        'https://www.youtube.com/watch?v=IakpTI4MERY',
        'https://www.youtube.com/watch?v=noZPPRnkatU',
        'https://www.youtube.com/watch?v=awp0vfmGKd8',
        'https://www.youtube.com/watch?v=-t1mWIf2Mw0',
        'https://www.youtube.com/watch?v=z26FbYVwwi4',
        'https://www.youtube.com/watch?v=JVBMh2_MzTI',
        'https://www.youtube.com/watch?v=mXR2-DMcEn4',
        'https://www.youtube.com/watch?v=CHeu7TIJGhY',
        'https://www.youtube.com/watch?v=OeYAhYTpMsQ',
        'https://www.youtube.com/watch?v=1vJRtsNcShY',
        'https://www.youtube.com/watch?v=VF9ecabaV2Q',
        'https://www.youtube.com/watch?v=Yd4ZAsy4Vfg',
        'https://www.youtube.com/watch?v=l0tuFrIcbQ0',
        'https://www.youtube.com/watch?v=39QGwB1cCw0',
        'https://www.youtube.com/watch?v=LZMSBDLikx0',
        'https://www.youtube.com/watch?v=y8RdG1Ek2Hc',
        'https://www.youtube.com/watch?v=4dH222I_sDM',
        'https://www.youtube.com/watch?v=E_6okj_TMB4',
        'https://www.youtube.com/watch?v=Zt3LPR6x7gU',
        'https://www.youtube.com/watch?v=BQ6Nf88CYX0',
        'https://www.youtube.com/watch?v=Nucy705UKt8',
        'https://www.youtube.com/watch?v=oQDIY8pWpo0',
        'https://www.youtube.com/watch?v=cWKVJajprxU',
        'https://www.youtube.com/watch?v=302jH6dbZ_c',
        'https://www.youtube.com/watch?v=HrIlCTosUYs',
        'https://www.youtube.com/watch?v=HBxJJ0IDsjc',
        'https://www.youtube.com/watch?v=e3A7wyAea4g',
        'https://www.youtube.com/watch?v=qXns5mSu4uI',
        'https://www.youtube.com/watch?v=RbRFZUl9950',
        'https://www.youtube.com/watch?v=Fpie47t8gIE',
        'https://www.youtube.com/watch?v=7U0eINq1i_Y',
        'https://www.youtube.com/watch?v=fGKyu_inEa8',
        'https://www.youtube.com/watch?v=HS5K5JFvTbs',
        'https://www.youtube.com/watch?v=nOTzWSXrgHc',
        'https://www.youtube.com/watch?v=LwY5Wb8uizU',
        'https://www.youtube.com/watch?v=8ZauD9qQKtA',
        'https://www.youtube.com/watch?v=yePfeID5oUs',
        'https://www.youtube.com/watch?v=xNPSgxzLmFA',
        'https://www.youtube.com/watch?v=cQQH1DLuKQM',
        'https://www.youtube.com/watch?v=BUWLDneSXvk',
        'https://www.youtube.com/watch?v=8Qx-MooEXsg',
        'https://www.youtube.com/watch?v=Mssv3JGgQQM',
        'https://www.youtube.com/watch?v=X0nglNoA5sM',
        'https://www.youtube.com/watch?v=As6ZbslSuD0',
        'https://www.youtube.com/watch?v=XEpoCeABVj4',
        'https://www.youtube.com/watch?v=eSmRLvUQ1HQ',
        'https://www.youtube.com/watch?v=OPjEoE7hwLM',
        'https://www.youtube.com/watch?v=playlists',
        'https://www.youtube.com/watch?v=XFdRiC_mOEM',
        'https://www.youtube.com/watch?v=7jeJqgOices',
        'https://www.youtube.com/watch?v=kdJZefm6fuk',
        'https://www.youtube.com/watch?v=O_8zsnSCqR0',
        'https://www.youtube.com/watch?v=Vr0YE3AhZBM',
        'https://www.youtube.com/watch?v=ui6zakUJ0WE',
        'https://www.youtube.com/watch?v=VC78kOjOYEQ',
        'https://www.youtube.com/watch?v=Knn_kobNyWw',
        'https://www.youtube.com/watch?v=SrqY4M5Vuqc',
        'https://www.youtube.com/watch?v=Jpa7jtFBu2w',
        'https://www.youtube.com/watch?v=pzrXsMovuOs',
        'https://www.youtube.com/watch?v=fvIPZe_UQQ8',
        'https://www.youtube.com/watch?v=6FtGCNqyf30',
        'https://www.youtube.com/watch?v=Bj6bap7UVJw',
        'https://www.youtube.com/watch?v=NOeD1gT33pg',
        'https://www.youtube.com/watch?v=Nrq_18-5JxM',
        'https://www.youtube.com/watch?v=DIfEZka5vUA',
        'https://www.youtube.com/watch?v=iZQLFr7EpCY',
        'https://www.youtube.com/watch?v=esOYul7Nw24',
        'https://www.youtube.com/watch?v=c7pYVpakpGA',
        'https://www.youtube.com/watch?v=3MpM78zEYaU',
        'https://www.youtube.com/watch?v=3iCpl5-8hcU',
        'https://www.youtube.com/watch?v=P2UZGDBiBg8',
        'https://www.youtube.com/watch?v=tZ0cOp7buRg',
        'https://www.youtube.com/watch?v=BMZVwGmGZNM',
        'https://www.youtube.com/watch?v=MPY8-yLrcNo',
        'https://www.youtube.com/watch?v=aaDXJnBtQEs',
        'https://www.youtube.com/watch?v=h1X5MX6rXb0',
        'https://www.youtube.com/watch?v=ZGcDmxfIAhI',
        'https://www.youtube.com/watch?v=TnnSSEEGBYo',
        'https://www.youtube.com/watch?v=dsCiQnZCX1g',
        'https://www.youtube.com/watch?v=lrQoWm9luvY',
        'https://www.youtube.com/watch?v=_PBkch91plA',
        'https://www.youtube.com/watch?v=9_5cdivmE98',
        'https://www.youtube.com/watch?v=rxhm0dBzjGA',
        'https://www.youtube.com/watch?v=a7B1trkVODE',
        'https://www.youtube.com/watch?v=SSifhJ2LMR0',
        'https://www.youtube.com/watch?v=391QsG7ZLxg',
        'https://www.youtube.com/watch?v=SAndNiSuDNI',
        'https://www.youtube.com/watch?v=MfQI_LIk-I0',
        'https://www.youtube.com/watch?v=1kWx-Q1jhhw',
        'https://www.youtube.com/watch?v=EnLPq16AmW0',
        'https://www.youtube.com/watch?v=txH7fXuR9F4',
        'https://www.youtube.com/watch?v=OuI4OyqXgqc',
        'https://www.youtube.com/watch?v=z5Ud8dx1Xb4',
        'https://www.youtube.com/watch?v=x8JG452paYI',
        'https://www.youtube.com/watch?v=gy9cX5Ey0TQ',
        'https://www.youtube.com/watch?v=cOiV02AeQfA',
        'https://www.youtube.com/watch?v=DuoUXSt4zSA',
        'https://www.youtube.com/watch?v=9nRqZ6ePbCE',
        'https://www.youtube.com/watch?v=8Yoi1pf1Mg8',
        'https://www.youtube.com/watch?v=RqwtRWZg1gw',
        'https://www.youtube.com/watch?v=uzLsSczGixA',
        'https://www.youtube.com/watch?v=rCnXTv-vpec',
        'https://www.youtube.com/watch?v=6ONKcJJCJWA',
        'https://www.youtube.com/watch?v=XJVSxzc61pw',
        'https://www.youtube.com/watch?v=jpOuGNr382s',
        'https://www.youtube.com/watch?v=wQ9NivK56vM',
        'https://www.youtube.com/watch?v=HDrJWC2F4xE',
        'https://www.youtube.com/watch?v=AXKqg-vhiFU',
        'https://www.youtube.com/watch?v=7wmjwo5fvmc',
        'https://www.youtube.com/watch?v=fQis8N2Yedw',
        'https://www.youtube.com/watch?v=aU0JTx4qv_Y',
        'https://www.youtube.com/watch?v=gmb-DAxevyI',
        'https://www.youtube.com/watch?v=0JDMdJ2OUr8',
        'https://www.youtube.com/watch?v=V3xTLxG0F7E',
        'https://www.youtube.com/watch?v=cfXxsXn3Nh4',
        'https://www.youtube.com/watch?v=8FmCrndY1z8',
        'https://www.youtube.com/watch?v=mYD1BhTzVrQ',
        'https://www.youtube.com/watch?v=czQhEhsNofw',
        'https://www.youtube.com/watch?v=h-6fGRYlHDo',
        'https://www.youtube.com/watch?v=ZrSfutgouC8',
        'https://www.youtube.com/watch?v=xtqk8TiGEqU',
        'https://www.youtube.com/watch?v=1dOo7YvESHM',
        'https://www.youtube.com/watch?v=Fd7l4c1LixQ',
        'https://www.youtube.com/watch?v=1-FlrN7gOMU',
        'https://www.youtube.com/watch?v=ngqMAZf8OGk',
        'https://www.youtube.com/watch?v=EYwUmUVSeWQ',
        'https://www.youtube.com/watch?v=-tB1Iq-5udo',
        'https://www.youtube.com/watch?v=_p_XyWnvHew',
        'https://www.youtube.com/watch?v=uSPAJNnhC3E',
        'https://www.youtube.com/watch?v=h_rXTHkRE4E',
        'https://www.youtube.com/watch?v=FuNVWrFSKYw',
        'https://www.youtube.com/watch?v=pvlnqvGqARs',
        'https://www.youtube.com/watch?v=A1LCDBdhpU0',
        'https://www.youtube.com/watch?v=rYr5ecqFhE0',
        'https://www.youtube.com/watch?v=pUq1y7G9_-o',
        'https://www.youtube.com/watch?v=Ysm7gH_zz1Q',
        'https://www.youtube.com/watch?v=dQiU4FIdSDo',
        'https://www.youtube.com/watch?v=0VjLLj0YJsM',
        'https://www.youtube.com/watch?v=ZlZUmgnQ4KU',
        'https://www.youtube.com/watch?v=cciDLMi80hA',
        'https://www.youtube.com/watch?v=TKbuRykv96M',
        'https://www.youtube.com/watch?v=3AknvPuESlk',
        'https://www.youtube.com/watch?v=V8LNr9MfYOk',
        'https://www.youtube.com/watch?v=J2xcOgngUsg',
        'https://www.youtube.com/watch?v=HGEfDvoyL1I',
        'https://www.youtube.com/watch?v=9Y7x2JBwI7o',
        'https://www.youtube.com/watch?v=nMEZIS-hzn0',
        'https://www.youtube.com/watch?v=_l5f_m_xNd0',
        'https://www.youtube.com/watch?v=CMUQbMxL47o',
        'https://www.youtube.com/watch?v=1Igi1shxE5w',
        'https://www.youtube.com/watch?v=hbKdirnPWs0',
        'https://www.youtube.com/watch?v=mFxD69H_cxM',
        'https://www.youtube.com/watch?v=JcxShS4bZBE',
        'https://www.youtube.com/watch?v=a684LKHaCU8',
        'https://www.youtube.com/watch?v=Uv13Fj8SFVc',
        'https://www.youtube.com/watch?v=eY2jeE5sgdw',
        'https://www.youtube.com/watch?v=mf4UAFZMiiM',
        'https://www.youtube.com/watch?v=VnRzvloJYqI',
        'https://www.youtube.com/watch?v=Q5hvshDGrwk',
        'https://www.youtube.com/watch?v=iKGNM1uJdL8',
        'https://www.youtube.com/watch?v=fbiDwUvTeMI',
        'https://www.youtube.com/watch?v=_BJnhd9prb8',
        'https://www.youtube.com/watch?v=luuV_Ia2ZLo',
        'https://www.youtube.com/watch?v=7uAhcGiFTtQ',
        'https://www.youtube.com/watch?v=povBP41VC2U',
        'https://www.youtube.com/watch?v=TQyuGhMT8uI',
        'https://www.youtube.com/watch?v=CxPyA74p70g',
        'https://www.youtube.com/watch?v=0RCu-ckuTqs',
        'https://www.youtube.com/watch?v=IeKkJWhyH2Q',
        'https://www.youtube.com/watch?v=QhPimqmoFaI',
        'https://www.youtube.com/watch?v=7yYHJnAYfwI',
        'https://www.youtube.com/watch?v=T6bq_Hz4I9Q',
        'https://www.youtube.com/watch?v=bpGUSDQ_E4U',
        'https://www.youtube.com/watch?v=CuCf8e7VSXY',
        'https://www.youtube.com/watch?v=bBOTqj-IjRc',
        'https://www.youtube.com/watch?v=i6V-e9od9FM',
        'https://www.youtube.com/watch?v=buBh4mMKbJc',
        'https://www.youtube.com/watch?v=x59rfJNXHSE',
        'https://www.youtube.com/watch?v=mY8GTkjMyDQ',
        'https://www.youtube.com/watch?v=K0vBV_Dxyl0',
        'https://www.youtube.com/watch?v=JDwRRSlvghE',
        'https://www.youtube.com/watch?v=zLwf8C_43vQ',
        'https://www.youtube.com/watch?v=7fWcLG8kvlE',
        'https://www.youtube.com/watch?v=fRX0DFkjUu4',
        'https://www.youtube.com/watch?v=Se8IZbgk0s0',
        'https://www.youtube.com/watch?v=lO_rzZJOw1o',
        'https://www.youtube.com/watch?v=qaFn0VdtFN8',
        'https://www.youtube.com/watch?v=OiqQBt0vnHw',
        'https://www.youtube.com/watch?v=mQrLvPP9YqY',
        'https://www.youtube.com/watch?v=krCjsxDhNyM',
        'https://www.youtube.com/watch?v=rnHixxhRWig',
        'https://www.youtube.com/watch?v=o4mgsfGHJy8',
        'https://www.youtube.com/watch?v=Y2JB5eKB7-Q',
        'https://www.youtube.com/watch?v=fPrq1sHu8ug',
        'https://www.youtube.com/watch?v=VqrEsBulM3M',
        'https://www.youtube.com/watch?v=o12k9gWuAYU',
        'https://www.youtube.com/watch?v=IUo3FDpNhI8',
        'https://www.youtube.com/watch?v=xFsY3vvvy_c',
        'https://www.youtube.com/watch?v=NotBXS9Cbzo',
        'https://www.youtube.com/watch?v=Z2qBcmM5GuU',
        'https://www.youtube.com/watch?v=733aQNIFNOc',
        'https://www.youtube.com/watch?v=Y4e20C-5rrE',
        'https://www.youtube.com/watch?v=6fubAEhWIoQ',
        'https://www.youtube.com/watch?v=U0Go67NZo9g',
        'https://www.youtube.com/watch?v=XocFfoFQYPc',
        'https://www.youtube.com/watch?v=j5Ut1huRaxE',
        'https://www.youtube.com/watch?v=m1kXKVaWoDo',
        'https://www.youtube.com/watch?v=Y6BfzOZsAq0',
        'https://www.youtube.com/watch?v=vfek32apijM',
        'https://www.youtube.com/watch?v=y2oxOv-OOkA',
        'https://www.youtube.com/watch?v=jN3cfc4V3F8',
        'https://www.youtube.com/watch?v=LmFiMdMoKac',
        'https://www.youtube.com/watch?v=cedeSJfCQZY',
        'https://www.youtube.com/watch?v=2ZK1AeP5bFE',
        'https://www.youtube.com/watch?v=VNpv55VA3oI',
        'https://www.youtube.com/watch?v=298ixmplajc',
        'https://www.youtube.com/watch?v=StIAvI4_X6w',
        'https://www.youtube.com/watch?v=NPAQSTyjmSA',
        'https://www.youtube.com/watch?v=LmOcJfV0hL0',
        'https://www.youtube.com/watch?v=6kyruIyhdME',
        'https://www.youtube.com/watch?v=T51rzaJy9kE',
        'https://www.youtube.com/watch?v=4fryJq-36QU',
        'https://www.youtube.com/watch?v=JxFq7d5fy-k',
        'https://www.youtube.com/watch?v=Rbq8oQkqeEY',
        'https://www.youtube.com/watch?v=4Rh3hvSE-14',
        'https://www.youtube.com/watch?v=CyjDNsFd8XE',
        'https://www.youtube.com/watch?v=ecgozwtirPk',
        'https://www.youtube.com/watch?v=4ORTDoXnRUI',
        'https://www.youtube.com/watch?v=ioG1kxVRwgA',
        'https://www.youtube.com/watch?v=QuVCAz8Kr8I',
        'https://www.youtube.com/watch?v=iprtyc',
        'https://www.youtube.com/watch?v=Xm8_8domqus',
        'https://www.youtube.com/watch?v=DqDRTTUGJYQ',
        'https://www.youtube.com/watch?v=cf436MP1GVc',
        'https://www.youtube.com/watch?v=vkygtyi8cds',
        'https://www.youtube.com/watch?v=b2MgOexJn0M',
        'https://www.youtube.com/watch?v=0b0GnqSbsFY',
        'https://www.youtube.com/watch?v=pmhY9z3DFj8',
        'https://www.youtube.com/watch?v=ofREiRYLZqE',
        'https://www.youtube.com/watch?v=vTEfRbqHXzc',
        'https://www.youtube.com/watch?v=-HGxyuYwzLc',
        'https://www.youtube.com/watch?v=83HeOjQpaio',
        'https://www.youtube.com/watch?v=4wGb682AjH8',
        'https://www.youtube.com/watch?v=5_Z4I51dKBY',
        'https://www.youtube.com/watch?v=GdP7MmR5iQ4',
        'https://www.youtube.com/watch?v=QkUcTcKZX5Y',
        'https://www.youtube.com/watch?v=BRZ0KvyVjjk',
        'https://www.youtube.com/watch?v=IbjvMU-_fHY',
        'https://www.youtube.com/watch?v=sEalmsaEYao',
        'https://www.youtube.com/watch?v=-0f0kaOcAkM',
        'https://www.youtube.com/watch?v=m1nP9M4GXKU',
        'https://www.youtube.com/watch?v=Eh7C9odfGf8',
        'https://www.youtube.com/watch?v=B6iP-4YkN8c',
        'https://www.youtube.com/watch?v=73hW00zBEwE',
        'https://www.youtube.com/watch?v=sjoHsim8Y-k',
        'https://www.youtube.com/watch?v=4pmHoMKo41s',
        'https://www.youtube.com/watch?v=iAQ_AU_FFlk',
        'https://www.youtube.com/watch?v=7E0t13c6XNE',
        'https://www.youtube.com/watch?v=U9n5DxQt3zw',
        'https://www.youtube.com/watch?v=4zzLeGWTGGc',
        'https://www.youtube.com/watch?v=HwaV577kMsQ',
        'https://www.youtube.com/watch?v=vQggXrliiaw',
        'https://www.youtube.com/watch?v=NFYXeRs-PVY',
        'https://www.youtube.com/watch?v=79IMms2GP3s',
        'https://www.youtube.com/watch?v=RG5-fYS9RLo',
        'https://www.youtube.com/watch?v=u3gq9Gmrb6Y',
        'https://www.youtube.com/watch?v=EWIpFncYefc',
        'https://www.youtube.com/watch?v=TMdU4mNOT9Q',
        'https://www.youtube.com/watch?v=jvK4pKMfMg0',
        'https://www.youtube.com/watch?v=yRSkWSOfsik',
        'https://www.youtube.com/watch?v=HXehwomWIno',
        'https://www.youtube.com/watch?v=91emOtFIEeY',
        'https://www.youtube.com/watch?v=Z5YCDoGwkc0',
        'https://www.youtube.com/watch?v=nUds_3IE64M',
        'https://www.youtube.com/watch?v=L1F8QMHVdiE',
        'https://www.youtube.com/watch?v=_4Vp_JbZEE0',
        'https://www.youtube.com/watch?v=3lVq14r57c0',
        'https://www.youtube.com/watch?v=kTKIwDQ6dPY',
        'https://www.youtube.com/watch?v=l8lY1WAwSDo',
        'https://www.youtube.com/watch?v=9fVTR6p5X_Q',
        'https://www.youtube.com/watch?v=GXfFP9ge9gs',
        'https://www.youtube.com/watch?v=zoCtzXxVKaU',
        'https://www.youtube.com/watch?v=a5APXbEATDY',
        'https://www.youtube.com/watch?v=GrRpoZnq5nY',
        'https://www.youtube.com/watch?v=OIvNgpOzZiI',
        'https://www.youtube.com/watch?v=q0eL_wRhf94',
        'https://www.youtube.com/watch?v=ZmbyHZh4CiY',
        'https://www.youtube.com/watch?v=wrx2IuakWf4',
        'https://www.youtube.com/watch?v=_hE-0oXUo-A',
        'https://www.youtube.com/watch?v=26wA6_dIx5I',
        'https://www.youtube.com/watch?v=sON-idEa4pU',
        'https://www.youtube.com/watch?v=cAueyi4I4Wo',
        'https://www.youtube.com/watch?v=x3cdwcoKLBE',
        'https://www.youtube.com/watch?v=6f_JbACf5Ps',
        'https://www.youtube.com/watch?v=YSxSILvlP8A',
        'https://www.youtube.com/watch?v=XocqeY6E4eA',
        'https://www.youtube.com/watch?v=brsB4z6vTzM',
        'https://www.youtube.com/watch?v=QEIdcc2ElIY',
        'https://www.youtube.com/watch?v=fd7LVFmx2Pc',
        'https://www.youtube.com/watch?v=N_zIdmGpUYo',
        'https://www.youtube.com/watch?v=z0E7gwtYMFg',
        'https://www.youtube.com/watch?v=kO2e0HON4-o',
        'https://www.youtube.com/watch?v=VQ86UsbJaFg',
        'https://www.youtube.com/watch?v=A2o3zZwlomI',
        'https://www.youtube.com/watch?v=MiNbTWits00',
        'https://www.youtube.com/watch?v=v77ahFkbcs0',
        'https://www.youtube.com/watch?v=lwbsiFnFCAc',
        'https://www.youtube.com/watch?v=Vjsf3bKv-K8',
        'https://www.youtube.com/watch?v=slJtN2la_8k',
        'https://www.youtube.com/watch?v=hfFStEsKQC8',
        'https://www.youtube.com/watch?v=dG4tub6Iq6c',
        'https://www.youtube.com/watch?v=S81EYJeGKKc',
        'https://www.youtube.com/watch?v=5uI-tuL1pn4',
        'https://www.youtube.com/watch?v=CIj9RRbP-hI',
        'https://www.youtube.com/watch?v=zpJof2fUVjw',
        'https://www.youtube.com/watch?v=aUHNuKGsNbs',
        'https://www.youtube.com/watch?v=52AxC3jF6rA',
        'https://www.youtube.com/watch?v=_uOtoA6sjc4',
        'https://www.youtube.com/watch?v=4mZauWGWbTI',
        'https://www.youtube.com/watch?v=N-g26diTysM',
        'https://www.youtube.com/watch?v=oCLOecOX_Fk',
        'https://www.youtube.com/watch?v=joLAkH0YAU8',
        'https://www.youtube.com/watch?v=i5gyQXSxwSc',
        'https://www.youtube.com/watch?v=zq3KfZgcOLk',
        'https://www.youtube.com/watch?v=MqthTXqY9Wo',
        'https://www.youtube.com/watch?v=TtilPaqxThw',
        'https://www.youtube.com/watch?v=cVqlpkrEQC8',
        'https://www.youtube.com/watch?v=ZfR6PCDdEGo',
        'https://www.youtube.com/watch?v=gwTP5Fn71j4',
        'https://www.youtube.com/watch?v=OZEuf_dQDi8',
        'https://www.youtube.com/watch?v=1r9ozONxb_g',
        'https://www.youtube.com/watch?v=PljDqBQY8xE',
        'https://www.youtube.com/watch?v=2LSnslNmgTo',
        'https://www.youtube.com/watch?v=0pPcJ_nLWM0',
        'https://www.youtube.com/watch?v=8Xaz0pG_R54',
        'https://www.youtube.com/watch?v=5JqWvH-xQxk',
        'https://www.youtube.com/watch?v=5BSohQCAkGI',
        'https://www.youtube.com/watch?v=AM8lZ4Rijac',
        'https://www.youtube.com/watch?v=7aqpfyiwcBQ',
        'https://www.youtube.com/watch?v=NSeu8V-e7IU',
        'https://www.youtube.com/watch?v=8HL9KUpfRtY',
        'https://www.youtube.com/watch?v=bJOB95a1MjY',
        'https://www.youtube.com/watch?v=5iIu4SRkaHA',
        'https://www.youtube.com/watch?v=t4mLUXxlLmQ',
    ]
    await check_videos_for_transcripts(youtube_links)
    return
    await clear_supabase_table(all_config["General"]["dataset_name_pages"])
    await clear_supabase_table(all_config["General"]["dataset_name_videos"])
    await start_crawling(["All"])
    
##############################################################################
if __name__ == "__main__":
    asyncio.run(main())

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
    await clear_supabase_table(all_config["General"]["dataset_name_pages"])
    await clear_supabase_table(all_config["General"]["dataset_name_videos"])
    await start_crawling(["All"])
    
##############################################################################
if __name__ == "__main__":
    asyncio.run(main())

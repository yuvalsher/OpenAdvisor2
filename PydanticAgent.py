from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic import Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import Annotated, List, Dict
from googleapiclient.discovery import build

from OpenAI_Assistant import OpenAIAssistant
from config import all_config
from utils import extract_html_body, get_html_from_url, get_md_from_html, load_json_file, get_hebert_embedding, get_longhero_embedding

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)
logfire.configure(send_to_logfire='if-token-present')
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")
# Build the service for Google Custom Search
google_search_service = build("customsearch", "v1", developerKey=google_api_key)

@dataclass
class PydanticAIDeps:
    supabase: Annotated[Client, "The Supabase client"]
    openai_client: Annotated[AsyncOpenAI, "The OpenAI client"]
    uploaded_files: Annotated[List[Dict[str, str]], "A list of dictionaries with 'name' and 'content' keys for uploaded files"]

system_prompt = """
    You are an advanced AI assistant with vast knowledge of the Open University of Israel (האוניברסיטה הפתוחה), designed to provide helpful information.  
    Your primary function is to assist users by answering questions, offering explanations, and providing insights, adapting your communication style to suit different users and contexts. 
    Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.
    Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.
    When using RAG, you can also check the list of available documentation pages and retrieve the content of page(s).
    Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
    The language of most of the content is Hebrew. Hebrew names for elements such as course names, faculty names, study program names, etc. must be provided verbatim, exactly as given in the provided content (returned by the provided tools).
    The query dependencies in the run context may include the text contents of PDF files uploaded by the user. Use this content to answer the user query. When answering any user query, if there is any uploaded file content available in ctx.deps.uploaded_files, please consider it and attach its context to your answer.
    
    - **Your Task:**
        Your task is to analyze the user query and decide how it should be handled. 
        First identify if the question involves a specific study program or not. If it does, act on the first category below. If it doesn't, act on the second or third category. 
        The query will fall into one of the following categories:

        - 1. Study Programs: Questions about specific study programs offered by the university. Study programs are a collection of requirements for eligibility for an academic degree. Study programs have a name in Hebrew, and a code which is a short (less than 10 characters) and unique string of uppercase English characters and optionally some numbers. Examples of study program codes include 'AF' and 'G188'. Study program details involve several sections, each of them can be a list of elective or required courses, with a requirement for minimum credit points. These queries should be handled using the a study program tool. The study program tool requires the study program code as input. If a study program name is provided in the user query, match the name to the list of study program names and codes. If the study program name is not provided in the query, you must ask the user for the study program name to identify the code. If you are unsure about the study program name, you can ask the user to approve your choice, or provide the study program name. Once you have the study program code, use the study program tool with the study program code and the question that the tool should answer. The tool does not have access to the chat history, so you must rephrase the question so that it contains all the relevant details from the chat history. If the query includes the contents of a PDF file, with the grade status of the student, provide that content to the study program tool as a parameter. 
        - 2. Course Details: Questions about specific university courses. Use the courses tools to answer these questions. Course details include course ID, name, URL, credits, classification, dependent courses, course overlaps, course overview, and available semesters. Always provide the url of the course page in the response.
        - 3. General University Information: General questions about studying at the university. Use the GetRelevantContent tool to search for relevant information from the university website. Always provide the source url of the information in the response.
        
    - **Expected Response Format:**
        If the user query is unclear or missing necessary information, such as the name of the study program, the response should be a clarifying question for gathering the required details from the user.
        Otherwise the response should be a detailed answer to the user question.

    - **Role and Tools:**
        - Utilize various tools to provide concise and accurate answers.
        - If you initially suspect that the query involves a study program, but no you cannot match it to any provided study programs, use GetRelevantContent tool to search for relevant information from the university website.
        - Access a comprehensive database of OUI courses through these tools to respond to user inquiries about course offerings and details.

    - **General Guidelines:**
        - Ensure all responses are clear, concise, and relevant.
        - Leverage the available tools effectively to provide accurate information.
        - Maintain the flow of conversation and context using the conversation memory.

    - **Course Structure:**
        Each course has a unique ID (course number), a unique name, and additional pertinent details.
        Utilize specific tools to retrieve essential course information and an overview that encompasses all relevant details.

        - Semesters:
            - Courses are offered in specific semesters:
                - '2025א' – First (winter) semester of 2025
                - '2025ב' – Second (spring) semester of 2025
                - '2025ג' – Third (summer) semester of 2025

        - Classifications:
            - Courses are classified under one or more departments ("סיווגים"), typically indicating the department and faculty offering the course.

        - Overlaps:
            - Some courses overlap with others. Use the 'overlap_courses' tool to list courses with full or partial overlaps.
            - For detailed information on overlapping courses, utilize the 'overlap_url' tool.
            - If a user inquires specifically about overlaps, provide the overlap URL.

        - Dependencies:
            - Courses may have dependencies that must be satisfied before enrollment:
                - **Condition Dependencies ("תנאי קבלה")**: Mandatory prerequisites without which registration fails.
                - **Required Dependencies ("ידע קודם דרוש")**: Courses that should be taken prior.
            - **Recommended Dependencies ("ידע קודם מומלץ")**: Courses that are advised to be taken beforehand.
            - Each dependency type has two associated tools:
                - One returns the textual description of the dependencies.
                - The other returns a list of courses that are dependencies.
            - Note that some dependent courses may have overlaps, requiring only a subset to be completed.

        - Similarity Check:
            - The tool `GetSimilarCourses` employs an embeddings vector search to identify courses similar to the input query text.
            - It returns a fixed number of results which may include irrelevant courses. Each result should be reviewed to ensure relevance.
    - **IMPORTANT:**
        - If the query includes the contents of a PDF file, with the grade status of the student, provide that content to the study program tool as a parameter.

    - **File Handling:**
        - You have access to uploaded files through ctx.deps.uploaded_files
        - Each file in uploaded_files is a dictionary with 'name' and 'content' keys
        - When answering questions, ALWAYS check ctx.deps.uploaded_files first if they contain relevant information
        - If a user asks about their grades or course status, the information should be in the uploaded files
"""

course_by_id = {}
course_by_name = {} 
course_data = load_json_file("all_courses.json", all_config["General"])
for course in course_data:
    course_by_id[course['course_id']] = course
    course_by_name[course['course_name']] = course

study_programs = {}
program_data = load_json_file("cs_study_programs.json", all_config["General"])
for program in program_data:
    study_programs[program['name']] = program['code']

# Create agent for OpenAI Assistant for Study Programs
study_programs_assistant = OpenAIAssistant(all_config["General"])
study_programs_assistant.init()


open_university_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

# ##############################################################################
# async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
#     """Get embedding vector from OpenAI."""
#     try:
#         response = await openai_client.embeddings.create(
#             model="text-embedding-3-small",
#             input=text
#         )
#         return response.data[0].embedding
#     except Exception as e:
#         print(f"Error getting embedding: {e}")
#         logfire.error('Error in get_embedding.', Exception = e)
#         return [0] * 1536  # Return zero vector on error


# ##############################################################################
# @open_university_expert.tool
# async def retrieve_relevant_web_pages(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
#     """
#     Retrieve relevant chunks of web pages based on the query with RAG.
    
#     Args:
#         ctx: The context including the Supabase client, the OpenAI client, and the uploaded files
#         user_query: The user's question or query
        
#     Returns:
#         A formatted string containing the top 5 most relevant chunks of web pages
#     """
#     try:        
#         # Get the embedding for the query
#         #query_embedding = get_hebert_embedding(user_query)
#         query_embedding = get_longhero_embedding(user_query)
        


#         # Query Supabase for relevant documents
#         result = ctx.deps.supabase.rpc(
#             'match_site_pages',
#             {
#                 'query_embedding': query_embedding,
#                 'match_count': 10,
#                 'filter': {'source': all_config["General"]["dataset_name_pages"]}
#             }
#         ).execute()
        
#         if not result.data:
#             logfire.info('Tool retrieve_relevant_web_pages: No relevant documentation found.', query = user_query)
#             return "No relevant documentation found."
            
#         logfire.info(f'Tool retrieve_relevant_web_pages: Found relevant documentation.', query = user_query, data = result.data)
#         # Format the results
#         formatted_chunks = []
#         for doc in result.data:
#             chunk_text = f"""
# # {doc['title']}

# {doc['content']}
# """
#             formatted_chunks.append(chunk_text)
            
#         # Join all chunks with a separator
#         return "\n\n---\n\n".join(formatted_chunks)
        
#     except Exception as e:
#         print(f"Error retrieving documentation: {e}")
#         logfire.error('Error in tool retrieve_relevant_web_pages.', Exception = e, query = user_query)
#         return f"Error retrieving documentation: {str(e)}"

# ##############################################################################
# @open_university_expert.tool
# async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
#     """
#     Retrieve a list of all available Pydantic AI documentation pages.
    
#     Args:
#         ctx: The context including the Supabase

#     Returns:
#         List[str]: List of unique URLs for all documentation pages
#     """
#     try:
#         # Query Supabase for unique URLs where source is pydantic_ai_docs
#         result = ctx.deps.supabase.from_('site_pages') \
#             .select('url') \
#             .eq('metadata->>source',  all_config["General"]["dataset_name_pages"]) \
#             .execute()
        
#         if not result.data:
#             logfire.info('Tool list_documentation_pages: No relevant documentation found.')
#             return []

#         # Extract unique URLs
#         urls = sorted(set(doc['url'] for doc in result.data))
#         logfire.info(f'Tool list_documentation_pages: Found {len(urls)} documentation pages.', data = urls)
#         return urls

#     except Exception as e:
#         print(f"Error retrieving documentation pages: {e}")
#         logfire.error('Error in tool list_documentation_pages.', Exception = e)
#         return []

# ##############################################################################
# @open_university_expert.tool
# async def get_full_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
#     """
#     Retrieve the full content of a specific documentation page by combining all its chunks.
    
#     Args:
#         ctx: The context including the Supabase client
#         url: The URL of the page to retrieve
        
#     Returns:
#         str: The complete page content with all chunks combined in order
#     """
#     try:
#         # Query Supabase for all chunks of this URL, ordered by chunk_number
#         result = ctx.deps.supabase.from_('site_pages') \
#             .select('title, content, chunk_number') \
#             .eq('url', url) \
#             .eq('metadata->>source',  all_config["General"]["dataset_name_pages"]) \
#             .order('chunk_number') \
#             .execute()
        
#         if not result.data:
#             logfire.info(f'Tool get_full_page_content: No content found for URL: {url}')
#             return f"No content found for URL: {url}"
            

#         # Format the page with its title and all chunks
#         page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
#         formatted_content = [f"# {page_title}\n"]
        
#         # Add each chunk's content
#         for chunk in result.data:
#             formatted_content.append(chunk['content'])
            
#         # Join everything together
#         final_result = "\n\n".join(formatted_content)
#         logfire.info(f'Tool get_full_page_content: Found content for URL: {url}', data = final_result)
#         return final_result
        

#     except Exception as e:
#         print(f"Error retrieving page content: {e}")
#         logfire.error('Error in tool get_full_page_content.', Exception = e)
#         return f"Error retrieving page content: {str(e)}"
    

##############################################################################
@open_university_expert.tool
async def retrieve_relevant_videos(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant chunks of YouTube video transcripts based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant chunks of video transcripts
    """
    try:
        # Get the embedding for the query
        #query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        query_embedding = get_longhero_embedding(user_query)
        

        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': all_config["General"]["dataset_name_videos"]}
            }
        ).execute()
        
        if not result.data:
            logfire.info('Tool retrieve_relevant_videos: No relevant documentation found.')
            return "No relevant documentation found."
            

        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        final_result = "\n\n---\n\n".join(formatted_chunks)
        logfire.info('Tool retrieve_relevant_videos: Found relevant documentation.', data = final_result)
        return final_result
        


    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        logfire.error('Error in tool retrieve_relevant_videos.', Exception = e)
        return f"Error retrieving documentation: {str(e)}"


##############################################################################
@open_university_expert.tool
async def get_study_program_code_from_name(ctx: RunContext[PydanticAIDeps], study_program_name: str) -> str:
    """Get the study program code from the study program name.
.            
    Args:
        study_program_name: The name of the study program to look up
    """

    if study_program_name not in study_programs:
        print(f"In Tool: Study program '{study_program_name[::-1]}' not found\n")
        logfire.info(f'Tool get_study_program_code_from_name: Study program {study_program_name} not found')
        return None
    

    result = study_programs[study_program_name]
    print(f"In Tool: Getting study program code for '{study_program_name[::-1]}': {result}\n")
    logfire.info(f'Tool get_study_program_code_from_name: Found study program code for {study_program_name}', data = result)
    return result
    

##############################################################################
@dataclass
class StudyProgram:
    program_name: str
    program_code: str

@open_university_expert.tool
async def get_list_of_study_program_names_and_codes(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """Get a list of all study program names.

    Returns:
        List[StudyProgram]: A list of study program names and codes.
    """

    result = [StudyProgram(name, code) for name, code in study_programs.items()]
    display_result = [f"    {program.program_name[::-1]}\n - {program.program_code[::-1]}" for program in result]
    print(f"In Tool: Getting list of study program names: \n{display_result}\n")
    logfire.info(f'Tool get_list_of_study_program_names_and_codes: Found list of study program names and codes', data = display_result)
    return result

##############################################################################
@open_university_expert.tool
async def get_answer_on_study_programs(ctx: RunContext[PydanticAIDeps], query_text: str, study_program_code: str) -> str:
    """Get an answer on study programs from the study programs assistant.
            
    Args:
        ctx: The context including uploaded files
        query_text: A question about a study program
        study_program_code: The code of the study program to answer the question
    """
    
    uploaded_files_content = []
    if ctx.deps.uploaded_files:
        for file in ctx.deps.uploaded_files:
            uploaded_files_content.append(file["content"])
    
    print(f"In Tool: Getting answer on study program {study_program_code} for '{query_text[::-1]}'\n")
    print(f"Number of uploaded files: {len(uploaded_files_content)}")
    print(f"Files content: {[content[:100] + '...' for content in uploaded_files_content]}")

    result = study_programs_assistant.do_query(
        query_text, 
        study_program_code, 
        uploaded_files_content
    )

    print(f"In Tool: Getting answer on study program {study_program_code} Returning: '{result[::-1]}'\n")
    logfire.info(f'Tool get_answer_on_study_programs: {study_program_code}', data = result)
    return result

##############################################################################
@open_university_expert.tool
async def get_course_id_from_name(ctx: RunContext[PydanticAIDeps], course_name: str) -> str:
    """Get the course ID from the course name.
    
    Args:
        course_name: The name of the course to look up
    """

    if course_name not in course_by_name:
        print(f"In Tool: Course '{course_name[::-1]}' not found")
        logfire.info(f'Tool get_course_id_from_name: Course {course_name} not found')
        return None
    

    result = course_by_name[course_name]['course_id']
    print(f"In Tool: Getting course ID for '{course_name[::-1]}': {result}\n")
    logfire.info(f'Tool get_course_id_from_name: Found course ID for {course_name}', data = result)
    return result

##############################################################################
@open_university_expert.tool
async def get_course_name_from_id(ctx: RunContext[PydanticAIDeps], course_id: str) -> str:
    """Get the course name from the course ID.
    
    Args:
        course_id: The ID of the course to look up
    """

    if course_id not in course_by_id:
        print(f"In Tool: Course {course_id} not found")
        logfire.info(f'Tool get_course_name_from_id: Course {course_id} not found')
        return None
    

    result = course_by_id[course_id]['course_name']
    print(f"In Tool: Getting course name for {course_id}: '{result[::-1]}'\n")
    logfire.info(f'Tool get_course_name_from_id: {course_id}', data = result)
    return result


##############################################################################
@dataclass
class CourseDetails:
    course_url: str
    course_id: str
    course_name: str
    credits: str
    classification: List[List[str]]
    condition_courses: List[str]
    required_deps_courses: List[str]
    recommended_deps_courses: List[str]
    overlap: List[str]
    text: List[str]
    footnotes: List[str]
    discontinued: bool
    overlap_url: str
    semesters: List[str]
    overlap_courses: List[str]
    condition_text: str = Field(default="")
    required_deps_text: str = Field(default="")
    recommended_deps_text: str = Field(default="")

@open_university_expert.tool
async def get_course_details_from_id(ctx: RunContext[PydanticAIDeps], course_id: str) -> CourseDetails:
    """Get the course details from the course ID.
    
    Args:
        course_id: The ID of the course to look up
    """

    if course_id not in course_by_id:
        print(f"In Tool: Course {course_id} not found")
        logfire.info(f'Tool get_course_details_from_id: Course {course_id} not found')
        return None
    

    result = course_by_id[course_id]
    print(f"In Tool: Getting course details for {course_id}: '{result['course_name'][::-1]}'\n")
    logfire.info(f'Tool get_course_details_from_id: {course_id}', data = result)
    return result


##############################################################################
@open_university_expert.tool
async def get_all_dependencies_courses_from_id(ctx: RunContext[PydanticAIDeps], course_id: str) -> List[str]:
    """Get the all the dependencies courses (condition, required and recommended) from the course ID.
    
    Args:
        course_id: The ID of the course to look up
    """
    if course_id not in course_by_id:
        print(f"In Tool: Course {course_id} not found\n")
        logfire.info(f'Tool get_all_dependencies_courses_from_id: Course {course_id} not found')
        return None
    

    deps = []   
    deps.extend(course_by_id[course_id]['condition_courses'])
    deps.extend(course_by_id[course_id]['required_deps_courses'])
    deps.extend(course_by_id[course_id]['recommended_deps_courses'])
    print(f"In Tool: Getting all dependencies courses for {course_id}: {deps}\n")
    logfire.info(f'Tool get_all_dependencies_courses_from_id: {course_id}', data = deps)
    return deps


##############################################################################
@open_university_expert.tool
async def get_course_overview_from_id(ctx: RunContext[PydanticAIDeps], course_id: str) -> str:
    """Get the course overview from the course ID. 
    
    Args:
        course_id: The ID of the course to look up
    """

    if course_id not in course_by_id:
        print(f"In Tool: Course {course_id} not found\n")   
        logfire.info(f'Tool get_course_overview_from_id: Course {course_id} not found')
        return None
    

    all_text = course_by_id[course_id]['text']
    # concatenate all_text from an array of strings to a single string
    result = ' '.join(all_text)
    print(f"In Tool: Getting course overview for {course_id}: '{result[::-1]}'\n")
    logfire.info(f'Tool get_course_overview_from_id: {course_id}', data = result)
    return result


##############################################################################
@open_university_expert.tool
async def attach_uploaded_files(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Attaches content from uploaded files to the user_query if available.
    This tool can be used for queries where additional context from uploaded files might be helpful.
    """
    if not ctx.deps.uploaded_files:
        logfire.info(f'Tool attach_uploaded_files: No uploaded files found')
        return user_query
    

    uploaded_contents = "\n".join(
        f"File: {file['name']}\nContent: {file['content']}" for file in ctx.deps.uploaded_files
    )
    print(f"In Tool: Attaching {len(ctx.deps.uploaded_files)} uploaded files to the query.")
    logfire.info(f'Tool attach_uploaded_files: {user_query}', data = uploaded_contents)
    # Append the uploaded file content to the query
    return f"{user_query}\n\nUploaded Files Context:\n{uploaded_contents}"


##############################################################################
@open_university_expert.tool
async def web_search(ctx: RunContext[PydanticAIDeps], query: str) -> List[str]:
    """Perform a web search using Google Custom Search API to find relevant information.
    
    Args:
        query: The search query to look up

    Returns:
        List[str]: A list of markdown documents with the cleaned up contents of search results.

    """
    try:
        # Run the synchronous request in an executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: google_search_service.cse().list(q=query, cx=google_cse_id, num=5).execute()
        )

        if not response or not response.get('items'):
            info_msg = f"In Tool: No relevant results found for query: '{query}'"
            print(info_msg)
            logfire.info('Tool web_search: No results found', query=query)
            return "No relevant results found."

        md_list = []
        semaphore = asyncio.Semaphore(10)
    
        async with semaphore:
            for item in response.get('items', []):
                url = item.get('link', '')
                md = get_page_content(url, ctx.deps.supabase)
#                html = await get_html_from_url(url)
#                html_body = extract_html_body(html)
#                md = await get_md_from_html(html_body, ctx.deps.openai_client)
                logfire.info('Tool web_search: Found a result', query=query, result=md)
                md_list.append(md)

        logfire.info('Tool web_search: Found results', query=query, result=md_list)
        return md_list

    except Exception as e:
        # Enhanced error handling for cases like HttpError 400 indicating invalid arguments.
        if hasattr(e, 'resp') and getattr(e.resp, 'status', None) == 400:
            error_details = "Request contains an invalid argument."
        else:
            error_details = str(e)
        logfire.error('Error in tool web_search', Exception=e)
        return f"Error performing web search: {error_details}"

##############################################################################
def get_page_content(url: str, supabasde_client: Client) -> str:
    # Query the 'site_pages' table
    try:
        response = (
            supabasde_client
            .table("site_pages")
            .select("chunk_number, content")
            .eq("url", url)
            .order("chunk_number")
            .execute()
        )

        # Extract the 'data' fields and concatenate them
        content_list = [item["content"] for item in response.data]
        concatenated_content = "".join(content_list)

        return concatenated_content
        
    except Exception as e:
        print(f"Error fetching data from Supabase: {e}")
        return None


from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List

from config import all_config

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
    You are an advanced AI assistant with vast knowledge of the Open University of Israel (האוניברסיטה הפתוחה), designed to provide helpful information.  
    Your primary function is to assist users by answering questions, offering explanations, and providing insights, adapting your communication style to suit different users and contexts. 
    Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.
    Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.
    When using RAG, you can also check the list of available documentation pages and retrieve the content of page(s).
    Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
    The language of most of the content is Hebrew. Hebrew names for elements such as course names, faculty names, study program names, etc. must be provided verbatim, exactly as given in the provided content (returned by the provided tools).
    The query may include the text contents of PDF files uploaded by the user. Use this content to answer the user query.
    
    - **Your Task:**
        Your task is to analyze the user query and decide how it should be handled. 
        First identify if the question involves a specific study program or not. If it does, act on the first category below. If it doesn't, act on the second or third category. 
        The query will fall into one of the following categories:

        - 1. Study Programs: Questions about specific study programs offered by the university. Study programs are a collection of requirements for eligibility for an academic degree. Study program details involve several sections, each of them can be a list of elective or required courses, with a requirement for minimum credit points. These queries should be handled using the a study program tool. The study program tool requires the study program code as input. If a study program name is provided in the user query, match the name to the list of study program names and codes. If the study program name is not provided in the query, you must ask the user for the study program name to identify the code. If you are unsure about the study program name, you can ask the user to approve your choice, or provide the study program name. Once you have the study program code, use the study program tool with the study program code and the question that the tool should answer. The tool does not have access to the chat history, so you must rephrase the question so that it contains all the relevant details from the chat history. If the query includes the contents of a PDF file, with the grade status of the student, provide that content to the study program tool as a parameter. 
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
"""

open_university_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

##############################################################################
async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

##############################################################################
@open_university_expert.tool
async def retrieve_relevant_web_pages(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant chunks of web pages based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant chunks of web pages
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': all_config["General"]["dataset_name_pages"]}
            }
        ).execute()
        
        if not result.data:
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
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

##############################################################################
@open_university_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source',  all_config["General"]["dataset_name_pages"]) \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

##############################################################################
@open_university_expert.tool
async def get_full_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source',  all_config["General"]["dataset_name_pages"]) \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
    
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
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
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
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"


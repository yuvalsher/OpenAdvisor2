import re
import os
from typing import List
import markdown
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

CHROMA_PATH = "OpenAdvisor2/kb/chroma"

MY_OPENAI_KEY = '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'

# System message to set the context or instructions for the assistant
system_message = (
    "You are an AI model serving as an academic advisor for the Open University of Israel (OUI). The name of the OUI in Hebrew is האוניברסיטה הפתוחה. "
    "Your primary role is to assist OUI students and prospective students by answering their questions related to studying at OUI. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Each piece of context include its source url. Always provide the sources of all the pieces of information you provide. "
    "\n\n---\n\n"
)

##############################################################################
def extract_course_numbers(query):
    # Regular expression to find all 5-digit numbers in the query
    course_number_pattern = r'\b\d{5}\b'
    
    # Find all matches
    return re.findall(course_number_pattern, query)

##############################################################################
def build_metadata_filter(course_numbers):
    # Build the metadata filter with an OR condition on course numbers
    if not course_numbers:
        return None
    
    if len(course_numbers) == 1:
        return {"course_number": course_numbers[0]}
    else:
        return {"course_number": {"$in": course_numbers}}

##############################################################################
# Define the prompt template to include system instructions, chat history, RAG chunks, and the user question
def prepare_prompt(user_input, chat_history, rag_chunks):
    # Function to format chat history and RAG chunks into text
    def format_history(history):
        return "\n".join([f"{msg['sender'].capitalize()}: {msg['message']}" for msg in history])

    #def format_rag_chunks(chunks):
    #    return "\n".join([f"- {chunk}" for chunk in chunks])

    # Prepare the input to the chain with system instructions, chat history, RAG chunks, and the user's question
    formatted_input = {
        "system": system_message,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        "chat_history": format_history(chat_history),
        "rag_chunks": rag_chunks,
        "user_input": user_input
    }

    # Create the prompt template
    prompt_template = PromptTemplate(
        input_variables=["system", "chat_history", "rag_chunks", "user_input"],
        template="{system}\n\n{chat_history}\n\n{rag_chunks}\n\n{user_input}"
    )

    # Ensure the objects are callable
    def prompt_callable(input_data):
        return prompt_template.format(**input_data)

    def llm_callable(input_text):
        return llm.generate(input_text)

    # Use RunnablePassthrough with callable functions
    #runnable_chain = RunnablePassthrough(func=prompt_callable, afunc=llm_callable)
    # The chain now includes the prompt template and the LLM to ensure the input is transformed into a response
    runnable_chain = prompt_template | llm  # Chain the prompt to the LLM directly

    return runnable_chain, formatted_input

##############################################################################
def retrieve_rag_chunks(query_text):
    global retriever
    
    def format_doc(doc):
            content = doc.page_content
            source = doc.metadata["source"]
            return f"{content}\nSource: {source}"

    def format_docs(docs):
        return "\n\n---\n\n".join(format_doc(doc) for doc in docs)
    
    # Retrieve relevant documents (chunks) from ChromaDB using the retriever
    retrieve_docs = (lambda x: x["input"]) | retriever

    # Extract course numbers from the query
    course_numbers = extract_course_numbers(query_text)
    print("Found course Numbers:", course_numbers)
    # Apply metadata filter if provided
    if course_numbers:
        # Build metadata filter for course numbers
        metadata_filter = build_metadata_filter(course_numbers)
        context_chunks = retrieve_docs.invoke({"input": query_text, "filter": metadata_filter})
    else:
        context_chunks = retrieve_docs.invoke({"input": query_text})
    
    print (f"Got {len(context_chunks)} RAG Chunks")

    # Format the retrieved documents (chunks) for the prompt
    formatted_context = format_docs(context_chunks)
    
    return formatted_context

##############################################################################
def initialize_rag(faculty_code):
    global db, llm, retriever, prompt_template
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = "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"

    # Init the ChromaDB.
    embedding_function = OpenAIEmbeddings()
    chroma_path = f"{CHROMA_PATH}_{faculty_code}"
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    llm = ChatOpenAI(model="gpt-4o-mini")
    retriever = db.as_retriever()

    # Init the prompt template
    prompt_template = PromptTemplate(
        input_variables=["system", "chat_history", "rag_chunks", "user_input"], 
        template="""
{system}

Chat History:
{chat_history}

RAG Chunks:
{rag_chunks}

User: {user_input}
Assistant:
"""
    )

##############################################################################
def format_response(response):
    """
    This function cleans up and formats the response.
    You can add more formatting steps as needed.
    """
    # Strip leading/trailing whitespace
    formatted_response = response.strip()

    # Optionally, replace multiple newlines with a single one
    formatted_response = formatted_response.replace("\n\n", "\n")

    # Add any other formatting tweaks here if necessary
    
    return formatted_response

##############################################################################
def format_markdown(response):
    # Render Markdown as HTML, if needed
    formatted_response = markdown.markdown(response.strip())
    return formatted_response

##############################################################################
def get_rag_response(user_input, chat_history):

    print("Initiating RAG")

    rag_chunks = retrieve_rag_chunks(user_input)

    (runnable_chain, formatted_input) = prepare_prompt(user_input, chat_history, rag_chunks)
    # Run the chain using the .invoke() method
    response = runnable_chain.invoke(formatted_input)

    # Print the response
    print("Returning LLM Response")

    answer = response.content
    return answer
    #return format_markdown(answer)

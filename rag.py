import re
import os
import logging
from typing import List
import markdown
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

class Rag:
    ##############################################################################
    def __init__(self):
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Set CHROMA_PATH relative to the script directory
        self.CHROMA_PATH = os.path.join(script_dir, "kb", "chroma")
        self.MY_OPENAI_KEY = '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'
        self.system_message = (
            "You are an AI model serving as an academic advisor for the Open University of Israel (OUI). The name of the OUI in Hebrew is האוניברסיטה הפתוחה. "
            "Your primary role is to assist OUI students and prospective students by answering their questions related to studying at OUI. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Each piece of context include its source url. Always provide the sources of all the pieces of information you provide. "
            "\n\n---\n\n"
        )
        self.db = None
        self.llm = None
        self.retriever = None
        self.prompt_template = None
        self.prompt_template_text="""
    {system}

    Chat History:
    {chat_history}

    RAG Chunks:
    {rag_chunks}

    User: {user_input}
    Assistant:
    """

    ##############################################################################
    def init(self, faculty_code):
        self.faculty_code = faculty_code
        self.embedding_function = OpenAIEmbeddings()
        
        # Define the path to the Chroma DB
        db_path = f"{self.CHROMA_PATH}_{faculty_code}"
        
        # Check if the directory exists
        if not os.path.exists(db_path):
            logging.error(f"Chroma DB directory does not exist: {db_path}")
            raise FileNotFoundError(f"Chroma DB directory not found: {db_path}")

        try:
            self.vectordb = Chroma(persist_directory=db_path, embedding_function=self.embedding_function)
            logging.info(f"Successfully opened Chroma DB at {db_path}")
            
            # Verify that the DB contains documents
            collection_size = self.vectordb._collection.count()
            if collection_size == 0:
                logging.warning(f"Chroma DB at {db_path} is empty")
            else:
                logging.info(f"Chroma DB contains {collection_size} documents")

        except Exception as e:
            logging.error(f"Failed to open Chroma DB: {str(e)}")
            raise

        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.retriever = self.vectordb.as_retriever()

        self.prompt_template = PromptTemplate(
            input_variables=["system", "chat_history", "rag_chunks", "user_input"], 
            template=self.prompt_template_text)

    ##############################################################################
    def get_rag_response(self, user_input, chat_history):
        print("Initiating RAG")
        rag_chunks = self._retrieve_rag_chunks(user_input)
        (runnable_chain, formatted_input) = self._prepare_prompt(user_input, chat_history, rag_chunks)
        response = runnable_chain.invoke(formatted_input)
        print("Returning LLM Response")
        answer = response.content
        return answer

    ##############################################################################
    def _extract_course_numbers(self, query):
        course_number_pattern = r'\b\d{5}\b'
        return re.findall(course_number_pattern, query)

    ##############################################################################
    def _build_metadata_filter(self, course_numbers):
        if not course_numbers:
            return None
        if len(course_numbers) == 1:
            return {"course_number": course_numbers[0]}
        else:
            return {"course_number": {"$in": course_numbers}}

    ##############################################################################
    def _prepare_prompt(self, user_input, chat_history, rag_chunks):
        def format_history(history):
            return "\n".join([f"{msg['sender'].capitalize()}: {msg['message']}" for msg in history])

        formatted_input = {
            "system": self.system_message,
            "chat_history": format_history(chat_history),
            "rag_chunks": rag_chunks,
            "user_input": user_input
        }

        runnable_chain = self.prompt_template | self.llm

        return runnable_chain, formatted_input

    ##############################################################################
    def _retrieve_rag_chunks(self, query_text):
        def format_doc(doc):
            content = doc.page_content
            source = doc.metadata["source"]
            return f"{content}\nSource: {source}"

        def format_docs(docs):
            return "\n\n---\n\n".join(format_doc(doc) for doc in docs)
        
        retrieve_docs = (lambda x: x["input"]) | self.retriever

        course_numbers = self._extract_course_numbers(query_text)
        print("Found course Numbers:", course_numbers)
        if course_numbers:
            metadata_filter = self._build_metadata_filter(course_numbers)
            context_chunks = retrieve_docs.invoke({"input": query_text, "filter": metadata_filter})
        else:
            context_chunks = retrieve_docs.invoke({"input": query_text})
        
        print(f"Got {len(context_chunks)} RAG Chunks")

        formatted_context = format_docs(context_chunks)
        
        return formatted_context

    ##############################################################################
    def _format_response(self, response):
        formatted_response = response.strip()
        formatted_response = formatted_response.replace("\n\n", "\n")
        return formatted_response

    ##############################################################################
    def _format_markdown(self, response):
        formatted_response = markdown.markdown(response.strip())
        return formatted_response

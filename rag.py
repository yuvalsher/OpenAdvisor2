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

class Rag:
    ##############################################################################
    def __init__(self):
        self.CHROMA_PATH = "OpenAdvisor2/kb/chroma"
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
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = "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"

        embedding_function = OpenAIEmbeddings()
        chroma_path = f"{self.CHROMA_PATH}_{faculty_code}"
        self.db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.retriever = self.db.as_retriever()

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


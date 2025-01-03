import re
import os
import logging
from typing import List
import markdown
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from AbstractLlm import AbstractLlm

class Rag(AbstractLlm):
    ##############################################################################
    def __init__(self, config):
        super().__init__(config)
        self.db = None
        self.llm = None
        self.retriever = None
        self.prompt_template = None
        self.msg_sender_field = self.config["msg_sender_field"]
        self.msg_text_field   = self.config["msg_text_field"]

        self.prompt_template_text = """
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

        # Use the system message from the config
        self.system_message = self.config["Rag_System_message"]
        
        # Define the path to the Chroma DB
        db_path = f"{self.config['Chroma_Path']}_{self.faculty_code}"
        
        # Check if the directory exists
        if not os.path.exists(db_path):
            logging.error(f"Chroma DB directory does not exist: {db_path}")
            raise FileNotFoundError(f"Chroma DB directory not found: {db_path}")

        try:
            # Initialize the embedding function with the API key
            self.embedding_function = OpenAIEmbeddings(
                model=self.config["embeddings"],
                openai_api_key=self.config["OPENAI_API_KEY"]
            )
            
            # Initialize Chroma with the embedding function
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

        # Initialize ChatOpenAI with the API key
        self.llm = ChatOpenAI(model=self.config["llm_model"], openai_api_key=self.config["OPENAI_API_KEY"])
        self.retriever = self.vectordb.as_retriever()

        self.prompt_template = PromptTemplate(
            input_variables=["system", "chat_history", "rag_chunks", "user_input"], 
            template=self.prompt_template_text)

    ##############################################################################
    def do_query(self, user_input: str, chat_history: list[dict], client_id: str = None) -> tuple[str, str]:
        """
        Process a query using RAG.
        
        Args:
            user_input: The user's query
            chat_history: The chat history used to maintain conversation context
            client_id: The client's unique identifier (not used in RAG but required for interface)
            
        Returns:
            tuple: (response_text, client_id)
        """
        response = self._do_rag_query(user_input, chat_history)
        return response, client_id

    ##############################################################################
    def reset_chat_history(self, client_id: str):
        """
        Reset chat history for a specific client.
        Note: RAG doesn't maintain persistent chat history, 
        but uses the history provided in each query.
        
        Args:
            client_id: The client's unique identifier
        """
        pass  # RAG uses chat history from parameters, no need to reset

    ##############################################################################
    def _extract_course_numbers(self, query):
        course_number_pattern = r'\b\d{5}\b'
        return re.findall(course_number_pattern, query)

    ##############################################################################
    def _do_rag_query(self, user_input: str, chat_history: list[dict] = None) -> str:
        # Get RAG chunks
        rag_chunks = self._retrieve_rag_chunks(user_input)
        
        # Prepare the prompt with chat history
        runnable_chain, formatted_input = self._prepare_prompt(user_input, chat_history or [], rag_chunks)
        
        # Get the response
        response = runnable_chain.invoke(formatted_input)
        
        # Format the response
        formatted_response = self._format_response(response.content)
        return formatted_response

    ##############################################################################
    def _build_metadata_filter(self, course_numbers):
        if not course_numbers:
            return None
        if len(course_numbers) == 1:
            return {"course_number": {"$eq": course_numbers[0]}}
        else:
            return {"course_number": {"$in": course_numbers}}

    ##############################################################################
    def _prepare_prompt(self, user_input, chat_history, rag_chunks):
        def _format_history(history):
            return "\n".join([f"{msg[self.msg_sender_field].capitalize()}: {msg[self.msg_text_field]}" for msg in history])

        formatted_input = {
            "system": self.system_message,
            "chat_history": _format_history(chat_history),
            "rag_chunks": rag_chunks,
            "user_input": user_input
        }

        runnable_chain = self.prompt_template | self.llm

        return runnable_chain, formatted_input

    ##############################################################################
    def _retrieve_rag_chunks(self, query_text):
        def format_doc(doc):
            content = doc['document']
            source = doc['metadata'].get('source', 'Unknown')
            return f"{content}\nSource: {source}"

        def format_docs(docs):
            return "\n\n---\n\n".join(format_doc(doc) for doc in docs)

        course_numbers = self._extract_course_numbers(query_text)
        print("Found course Numbers:", course_numbers)

        # Get the embedding for the query text
        query_embedding = self.vectordb._embedding_function.embed_query(query_text)

        if course_numbers:
            metadata_filter = self._build_metadata_filter(course_numbers)
            results = self.vectordb._collection.query(
                query_embeddings=[query_embedding],
                n_results=4,
                where=metadata_filter
            )
        else:
            results = self.vectordb._collection.query(
                query_embeddings=[query_embedding],
                n_results=4
            )

        context_chunks = results['documents'][0]
        metadatas = results['metadatas'][0]
        # Get the similarity scores - lower distances mean more similar documents
        # Chroma returns L2 distances between embeddings, ranging from 0 (identical) to 2 (completely different)
        distances = results['distances'][0]

        print(f"Got {len(context_chunks)} RAG Chunks")
        for chunk, metadata, distance in zip(context_chunks, metadatas, distances):
            print(f"Chunk metadata: {metadata}, Distance: {distance}")

        formatted_docs = [{'document': doc, 'metadata': meta} for doc, meta in zip(context_chunks, metadatas)]
        formatted_context = format_docs(formatted_docs)
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

    ##############################################################################
    def retrieve_rag_chunks_for_tool(self, query_text):

        def format_doc(doc):
            content = doc['document']
            for key, value in doc['metadata'].items():
                content += f"\n{key}: {value}"
            return content

        def format_docs(docs):
            return ", ".join(format_doc(doc) for doc in docs)

        # Get the embedding for the query text
        query_embedding = self.vectordb._embedding_function.embed_query(query_text)

        results = self.vectordb._collection.query(
            query_embeddings=[query_embedding],
            n_results=4
        )

        context_chunks = results['documents'][0]
        metadatas = results['metadatas'][0]
        # Get the similarity scores - lower distances mean more similar documents
        # Chroma returns L2 distances between embeddings, ranging from 0 (identical) to 2 (completely different)
        distances = results['distances'][0]

        print(f"Got {len(context_chunks)} RAG Chunks")
        for chunk, metadata, distance in zip(context_chunks, metadatas, distances):
            print(f"Chunk metadata: {metadata}, Distance: {distance}")

        formatted_docs = [{'document': doc, 'metadata': meta} for doc, meta in zip(context_chunks, metadatas)]
        formatted_context = format_docs(formatted_docs)
        return formatted_context


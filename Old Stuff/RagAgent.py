import os
import json
from langchain.agents import tool
from pydantic import BaseModel, Field
from typing import Optional, List

from AbstractAgent import AbstractAgent
from rag import Rag

class RagAgent(AbstractAgent):

    ##############################################################################
    def __init__(self, config, faculty_code: str):
        super().__init__(config)
        self.faculty_code = faculty_code
        
    ##############################################################################
    def _init_data(self):
        self.system_instructions = """
            You are an advanced AI assistant with vast knowledge of the Open University of Israel (האוניברסיטה הפתוחה), designed to provide helpful information.  
            Your primary function is to assist users by answering questions, offering explanations, and providing insights, adapting your communication style to suit different users and contexts. 
            The language of most of the content is Hebrew. Hebrew names for elements such as course names, faculty names, study program names, etc. must be provided verbatim, exactly as given in the provided content (returned by the provided tools).
            
            **Role and Tools:**
            - Utilize various tools to provide concise and accurate answers.
            - Access a comprehensive database of information taken from the Open University of Israel (OUI) website.

            **General Guidelines:**
            - Ensure all responses are clear, concise, and relevant.
            - Leverage the available tools effectively to provide accurate information.
            - Maintain the flow of conversation and context using the conversation memory.
        """

        self.rag = Rag(self.config)
        self.rag.init(self.faculty_code)

    ##############################################################################
    def get_system_instructions(self):
        return [self.system_instructions]

    ##############################################################################
    def _init_tools(self):

        ##############################################################################
        @tool("GetRelevantContent")
        def _get_relevant_content(query_text: str) -> str:
            """Find relevant content to the query text by using an embeddings vector search.
.            
            Args:
                query_text: A subject or topic to search for in the course overview
            """

            result = self.rag.retrieve_rag_chunks_for_tool(query_text)
            print(f"In Tool: Getting relevant content for faculty {self.faculty_code} and query {query_text}: \n{result}\n")
            return result
            
        ##############################################################################
        self.tools = [
                    _get_relevant_content,
                ]


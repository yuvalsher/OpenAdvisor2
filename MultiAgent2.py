from uuid import uuid4
from typing import Dict, Any
import io
import os
import PyPDF2
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from AbstractLlm import AbstractLlm
#from CoursesAgent import CoursesAgent
from OpenAI_Assistant2 import OpenAIAssistant
from RagAgent import RagAgent
from RouterAgent import RouterAgent
from utils import flip_by_line

##############################################################################
class MultiAgent2(AbstractLlm):
    ##############################################################################
    def __init__(self, config):
        # self.system_instructions = """
        #     You are an advanced AI assistant with vast knowledge of the Open University of Israel (האוניברסיטה הפתוחה), designed to provide helpful information.  
        #     Your primary function is to assist users by answering questions, offering explanations, and providing insights, adapting your communication style to suit different users and contexts. 
        #     The language of most of the content is Hebrew. Hebrew names for elements such as course names, faculty names, study program names, etc. must be provided verbatim, exactly as given in the provided content (returned by the provided tools).
        # """

        super().__init__(config)
        self.memories: Dict[str, ConversationBufferMemory] = {}

    ##############################################################################
    def init(self, faculty_code):
        self.llm = ChatOpenAI(model=self.config["llm_model"], temperature=0)

        # Create agent for courses
#        self.courses_agent_creator = CoursesAgent(self.config)
#        self.courses_agent_creator.init()

        # Create agent for general questions
        self.general_agent_creator = RagAgent(self.config, "OUI")
        self.general_agent_creator.init()

        # Create agent for CS faculty questions
#        self.cs_agent_creator = RagAgent(self.config, "CS")
#        self.cs_agent_creator.init()

        # Create agent for routing questions
        self.router_agent_creator = RouterAgent(self.config)
        self.router_agent_creator.init()

        # Create agent for routing questions
        self.assistant_agent = OpenAIAssistant(self.config)
        self.assistant_agent.init()
        
        self.example_raw = self.load_text_file("Student1-Raw.txt")
        self.example_formatted = self.load_text_file("Student1-Formatted.txt")

    ##############################################################################
    def _get_or_create_memory(self, client_id: str) -> ConversationBufferMemory:
        """Get existing memory for client_id or create new if doesn't exist."""
        try:
            if client_id in self.memories:
                return self.memories[client_id]
            else:
                memory = self.router_agent_creator.create_new_memory()
                self.memories[client_id] = memory
                return memory
        except Exception as e:
            print(f"Error in _get_or_create_memory: {e}")
            return None

    ##############################################################################
    def add_files_to_prompt(self, prompt: str, uploaded_files: list[Dict[str, str]]) -> str:
        """Add the content of the uploaded files to the prompt"""
        for file in uploaded_files:
            cleaned_text = self.rephrase_text(file['content'])
            prompt += f"\n\nHere is the content of the uploaded file {file['name']}:\n\n{cleaned_text}"
        return prompt

    ##############################################################################
    def load_text_file(self, file_name):
        # Read text file containing instructions
        text_path = os.path.join(self.config["DB_Path"], file_name)
        with open(text_path, "r", encoding='utf-8') as text_file:
            return text_file.read()

    ##############################################################################
    def rephrase_text(self, text, model="gpt-4o-mini", max_tokens=5000):
        """
        Rephrase the input text using OpenAI's API.
        
        Args:
        - text (str): The text to rephrase.
        - model (str): The OpenAI model to use (default is "gpt-4").
        - max_tokens (int): The maximum number of tokens for the summary (default is 300).
        
        Returns:
        - str: The summary of the input text.
        """
        prompt = f"""Please rephrase the following text using the input language. Keep all the relevant text, but clean it up and remove any irrelevant characters used for formatting. The file typically includes a table of courses with their details, and is formatted using ascii characters. The text is in Hebrew. The output format should be easily understood by an LLM. Start with the title of the document. In the example below, the text includes a list of completed courses with their grades, courses in progress without grades, and courses that are planned.
        Here is an example input text:
        {self.example_raw}
        Here is an example output text:
        {self.example_formatted}
        The text to rephrase is:
        {text}
        """
        # Set the OpenAI API key
        openai_epi_key = self.config["OPENAI_API_KEY"]
        client = OpenAI(api_key=openai_epi_key)
        
        try:
            # Call the OpenAI API to summarize the text
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{prompt}\n{text}"}
                ]
            )

            # Extract the summary from the response
            summary = response.choices[0].message.content.strip()
            return summary
        
        except Exception as e:
            return f"An error occurred: {e}"

    ##############################################################################
    def do_query(self, user_input: str, chat_history: list[dict], client_id: str = None, uploaded_files: list[Dict[str, str]] = None) -> tuple[str, str]:
        """
        Process a query using multiple agents.
        
        Args:
            user_input: The user's query
            chat_history: The chat history used to maintain conversation context
            client_id: The client's unique identifier (not used in RAG but required for interface)
            uploaded_files: List of paths to uploaded files
            
        Returns:
            tuple: (response_text, client_id)
        """

        print(f"Entering Multi-Agent: user_input: {user_input[::-1]}, Chat hiostory length: {len(chat_history)}, uploaded files: {len(uploaded_files)}")

        if client_id is None:
            client_id = str(uuid4())

        memory = self._get_or_create_memory(client_id)

        try:
            router_agent = self.router_agent_creator.get_agent()
            prompt = self.router_agent_creator.get_prompt() + user_input
            prompt = self.add_files_to_prompt(prompt, uploaded_files)
            router_agent.memory = memory
            print(f"Router agent prompt: {flip_by_line(prompt)}")
            response = router_agent.invoke({"input": prompt})
            router_response = response['output']
        except Exception as e:
            print(f"Error in router agent: {e}")
            router_response = f"Error in router agent: {e}"


        # result = None
        # agent = None
        # try:
        #     if router_response.lower().startswith("done - "):
        #         print(f"{router_response[::-1]}")
        #         result = router_response[len("done - "):]
        #     elif router_response.lower().startswith("question - "):
        #         print(f"{router_response[::-1]}")
        #         result = router_response[len("question - "):]
        #     elif router_response.lower().startswith("program "):
        #         program_code, question = router_response[len("program "):].split(": ", 1)
        #         print(f"Study Program - sending to assistant: {program_code}, Question: {question[::-1]}")
        #         result = self.assistant_agent.do_query(question, program_code, memory, client_id)
        #     else:
        #         print(f"No agent implemented yet for this query type: {router_response} - defaulting to general RAG agent")
        #         agent = self.general_agent_creator.get_agent()
        #         agent.memory = memory
        #         response = agent.invoke({"input": user_input})
        #         result = response['output']

        # except Exception as e:
        #     msg = f"Error processing router response {router_response}\nError: {e}"
        #     print(msg)
        #     result = msg

        return router_response, client_id

    ##############################################################################
    def reset_chat_history(self, client_id: str):
        """
        Reset chat history for a specific client.
        
        Args:
            client_id: The client's unique identifier
        """
        if client_id in self.memories:
            memory = self._create_new_memory(client_id)
            self.memories[client_id] = memory



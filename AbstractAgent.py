import os
import json
from abc import ABC, abstractmethod
from typing import List
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain import hub

class AbstractAgent(ABC):

    ##############################################################################
    def __init__(self, config):
        self.config = config

    ##############################################################################
    @abstractmethod
    def _init_tools(self):
        pass

    ##############################################################################
    @abstractmethod
    def _init_data(self):
        pass

    ##############################################################################
    def init(self):
        # Initialize a ChatOpenAI model
        self.llm = ChatOpenAI(model=self.config["llm_model"])
        
        self._init_tools()
        self._init_data()

    ##############################################################################
    def get_agent(self) -> AgentExecutor:
        """
        Create an AgentExecutor ready for invoking.
        Just set the chat history and call invoke()
        
        Returns:
            AgentExecutor: The agent ready for invoking
        """

        # Pull the prompt template from the hub
        self.prompt_template = hub.pull("hwchase17/openai-tools-agent")

        # Create the ReAct agent using the create_tool_calling_agent function
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt_template
        )

        # Create the agent executor with client-specific memory
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )

        return agent_executor
    
    ##############################################################################
    def _load_json_file(self, file_name: str):
        # Use the DB_Path from the config
        full_path = os.path.join(self.config["DB_Path"], file_name)
        if not os.path.exists(full_path):
            print(f"File {full_path} does not exist.")
            raise FileNotFoundError(f"File {full_path} does not exist.")
        
        try:    
            with open(full_path, "r", encoding='utf-8') as f:
                # read file contents into string    
                file_contents = f.read()
                # Remove BOM and direction marks
                file_contents = file_contents.strip('\ufeff\u200e\u200f')
                return json.loads(file_contents)
        except Exception as e:
            print(f"Error loading JSON file {file_name}: {str(e)}")
            raise e

    ##############################################################################
    @abstractmethod
    def get_system_instructions(self) -> List[str]:
        pass

    ##############################################################################
    def create_new_memory(self) -> str:
        """Create a new memory instance for a client and return its ID."""
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        system_instructions = self.get_system_instructions()
        for msg in system_instructions:
            memory.chat_memory.add_message(SystemMessage(content=msg))
        return memory



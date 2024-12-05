from typing import Dict, Any
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from AbstractLlm import AbstractLlm
from CoursesAgent import CoursesAgent
from RagAgent import RagAgent
from RouterAgent import RouterAgent

##############################################################################
class MultiAgent2(AbstractLlm):
    ##############################################################################
    def __init__(self, config):
        self.system_instructions = """
            You are an advanced AI assistant with vast knowledge of the Open University of Israel (האוניברסיטה הפתוחה), designed to provide helpful information.  
            Your primary function is to assist users by answering questions, offering explanations, and providing insights, adapting your communication style to suit different users and contexts. 
            The language of most of the content is Hebrew. Hebrew names for elements such as course names, faculty names, study program names, etc. must be provided verbatim, exactly as given in the provided content (returned by the provided tools).
        """

        super().__init__(config)
        self.memories: Dict[str, ConversationBufferMemory] = {}

    ##############################################################################
    def init(self, faculty_code):
        self.llm = ChatOpenAI(model=self.config["llm_model"], temperature=0)

        # Create agent for courses
        self.courses_agent_creator = CoursesAgent(self.config)
        self.courses_agent_creator.init()

        # Create agent for general questions
        self.general_agent_creator = RagAgent(self.config, "OUI")
        self.general_agent_creator.init()

        # Create agent for CS faculty questions
        self.cs_agent_creator = RagAgent(self.config, "CS")
        self.cs_agent_creator.init()

        # Create agent for routing questions
        self.router_agent_creator = RouterAgent(self.config)
        self.router_agent_creator.init()

    ##############################################################################
    def _create_new_memory(self, client_id: str = None) -> str:
        """Create a new memory instance for a client and return its ID."""
        if client_id is None:
            client_id = str(uuid4())
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        memory.chat_memory.add_message(SystemMessage(content=self.system_instructions))
        self.memories[client_id] = memory
        return client_id

    def _get_or_create_memory(self, client_id: str) -> ConversationBufferMemory:
        """Get existing memory for client_id or create new if doesn't exist."""
        try:
            if client_id not in self.memories:
                self._create_new_memory(client_id)
            return self.memories[client_id]
        except Exception as e:
            print(f"Error in _get_or_create_memory: {e}")
            return None

    ##############################################################################
    def do_query(self, user_input: str, chat_history: list[dict], client_id: str = None) -> tuple[str, str]:
        """
        Process a query using multiple agents.
        
        Args:
            user_input: The user's query
            chat_history: The chat history used to maintain conversation context
            client_id: The client's unique identifier (not used in RAG but required for interface)
            
        Returns:
            tuple: (response_text, client_id)
        """

        print(f"Entering Multi-Agent: user_input: {user_input[::-1]}")

        memory = self._get_or_create_memory(client_id)

        try:
            router_agent = self.router_agent_creator.get_agent()
            prompt = self.router_agent_creator.get_prompt() + user_input
            router_agent.memory = memory
            response = router_agent.invoke({"input": prompt})
            router_response = response['output']
        except Exception as e:
            print(f"Error in router agent: {e}")
            router_response = "Done - Error in router agent"


        result = None
        agent = None
        try:
            if router_response.startswith("Done - "):
                result = router_response.replace("Done - ", "")
            elif router_response.startswith("study_program - "):
                study_program = router_response.replace("study_program - ", "")

                result = f"No agent implemented yet for this query type: {router_response}"
            else:
                print(f"No agent implemented yet for this query type: {router_response}")
                agent = self.general_agent_creator.get_agent()

            if agent is not None:
                agent.memory = memory
                response = agent.invoke({"input": user_input})
                result = response['output']

        except Exception as e:
            print(f"Error in {router_response} agent: {e}")
            result = f"Error in {router_response} agent: {e}"

        return result, client_id

    ##############################################################################
    def reset_chat_history(self, client_id: str):
        """
        Reset chat history for a specific client.
        
        Args:
            client_id: The client's unique identifier
        """
        if client_id in self.memories:
            self.memories[client_id].clear()
            self.memories[client_id].chat_memory.add_message(
                SystemMessage(content=self.system_instructions)
            )


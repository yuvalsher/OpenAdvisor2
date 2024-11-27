from typing import Dict, Any
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, RouterChain
from langchain.chains.router import LLMRouterChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field

from AbstractLlm import AbstractLlm
from CoursesWithTools import CoursesWithTools
from CoursesAgent import CoursesAgent

##############################################################################
class MultiAgent(AbstractLlm):
    ##############################################################################
    def __init__(self, faculty_code, config):
        self.system_instructions = """
            You are an advanced AI assistant with vast knowledge of the Open University of Israel (האוניברסיטה הפתוחה), designed to provide helpful information.  
            Your primary function is to assist users by answering questions, offering explanations, and providing insights, adapting your communication style to suit different users and contexts. 
            The language of most of the content is Hebrew. Hebrew names for elements such as course names, faculty names, study program names, etc. must be provided verbatim, exactly as given in the provided content (returned by the provided tools).
        """

        super().__init__(faculty_code, config)
        self.courses_agent = CoursesAgent(config)
        self.memories: Dict[str, ConversationBufferMemory] = {}

    ##############################################################################
    def init(self):
        self.llm = ChatOpenAI(model=self.config["llm_model"], temperature=0)
        self.courses_agent.init()

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
        if client_id not in self.memories:
            self._create_new_memory(client_id)
        return self.memories[client_id]

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

        router_chain = self._build_router_agent(user_input)
        try:
            router_response = router_chain.invoke({"input": user_input})
            print(f"Router response: {router_response}")
        except Exception as e:
            print(f"Error in router chain: {e}")
            router_response = "general"

        try:
            if "course" in router_response:
                agent = self.courses_agent.get_agent()
                agent.memory = memory
                response = agent.invoke({"input": user_input})
            else:
                response = f"No agent implemented yet for this query type: {router_response}"
        except Exception as e:
            print(f"Error in {router_response} agent: {e}")
            response = f"Error in {router_response} agent: {e}"

        return response['output'], client_id

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

    ##############################################################################
    def _build_router_agent(self, user_input: str) -> str:
        # Define the RouterChain's prompt to decide which chain to use
        router_prompt = PromptTemplate(
            input_variables=["input"],
            template="""
                Given a user's question, categorize it into one of the following categories:

                - general: General questions about studying at the university.
                - faculty: Questions from students about the Computer Science faculty.
                - course: Questions on details of specific courses.
                - study_program: Questions regarding study programs.

                Provide only the category name that best fits the question.

                Question: {input}
                Category:"""
        )

        # Create the LLMRouterChain
        router_llm = ChatOpenAI(model=self.config["router_llm_model"], temperature=0)
        router_chain = router_prompt | router_llm | StrOutputParser()
        return router_chain


    
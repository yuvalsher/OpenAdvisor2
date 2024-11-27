from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Dict, Any

from AbstractLlm import AbstractLlm
from CoursesWithTools import CoursesWithTools

##############################################################################
class MultiAgent2(AbstractLlm):
    ##############################################################################
    def __init__(self, faculty_code, config):
        self.system_instructions = """
            You are an advanced AI assistant with vast knowledge of the Open University of Israel (האוניברסיטה הפתוחה), designed to provide helpful information.  
            Your primary function is to assist users by answering questions, offering explanations, and providing insights, adapting your communication style to suit different users and contexts. 
            The language of most of the content is Hebrew. Hebrew names for elements such as course names, faculty names, study program names, etc. must be provided verbatim, exactly as given in the provided content (returned by the provided tools).
        """

        super().__init__(faculty_code, config)
        self.courses_agent = CoursesWithTools(faculty_code, config)
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

        router = self._build_router_agent(user_input, chat_history, client_id)
        response = router.invoke({"query": user_input})

        # Output the result
        print(f"MultiAgent response: {response[::-1]}")

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
        if client_id in self.memories:
            self.memories[client_id].clear()
            self.memories[client_id].chat_memory.add_message(
                SystemMessage(content=self.system_instructions)
            )


    ##############################################################################
    def _build_router_agent(self, user_input: str, chat_history: list[dict], client_id: str = None) -> tuple[str, str]:

        # Define prompt templates for different query types
        general_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an assistant specialized in answering general questions about studying at the university."),
                ("human", "{query}"),
            ]
        )

        faculty_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an assistant specialized in answering questions about the Computer Science faculty."),
                ("human", "{query}"),
            ]
        )

        course_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an assistant specialized in providing details on specific courses."),
                ("human", "{query}"),
            ]
        )

        study_program_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an assistant specialized in answering questions regarding study programs, involving reasoning ."),
                ("human", "{query}"),
            ]
        )

        # Define the query classification template
        classification_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                ("human",
                """Given a user's question, categorize it into one of the following categories:
                    - general: General questions about studying at the university.
                    - faculty: Questions from students about the Computer Science faculty.
                    - course: Questions on details of specific courses.
                    - study_program: Questions regarding study programs.

                Provide only the category name that best fits the question.
                : {query}."""),
            ]
        )

        # Define the runnable branches for handling query
        branches = RunnableBranch(
            (
                lambda x: "study_program" in x,
                study_program_template | self.llm | StrOutputParser()  
            ),
            (
                lambda x: "faculty" in x,
                faculty_template | self.llm | StrOutputParser()  
            ),
            (
                lambda x: "course" in x,
                self.courses_agent.get_agent(client_id).invoke("{query}") | StrOutputParser() 
            ),
            general_template | self.llm | StrOutputParser()
        )

        # Create the classification chain
        classification_chain = classification_template | self.llm | StrOutputParser()

        # Combine classification and response generation into one chain
        chain = classification_chain | branches

        return chain
    
    
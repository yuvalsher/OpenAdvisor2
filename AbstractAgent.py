from abc import ABC, abstractmethod
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
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
            prompt=self.prompt_template,
        )

        # Create the agent executor with client-specific memory
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )

        return agent_executor
    

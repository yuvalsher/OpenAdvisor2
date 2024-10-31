# Docs: https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/
# 
import os
import json
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from AbstractLlm import AbstractLlm

##############################################################################
###### CoursesWithTools class  
##############################################################################
class CoursesWithTools(AbstractLlm):

    ##############################################################################
    def __init__(self, faculty_code, config):
        super().__init__(faculty_code, config)
        self.course_data = []
        self.course_by_id = {}
        self.course_by_name = {}

    ##############################################################################
    def _init_data(self):
        # Use the DB_Path from the config
        full_path = os.path.join(self.config["DB_Path"], "all_courses.json")

        # if the file does not exist - throw exception
        if not os.path.exists(full_path):
            print(f"File {full_path} does not exist.")
            raise FileNotFoundError(f"File {full_path} does not exist.")
            return
        
        with open(full_path, "r", encoding='utf-8') as f:
            self.course_data = json.load(f)

        for course in self.course_data:
            self.course_by_id[course['course_id']] = course
            self.course_by_name[course['course_name']] = course

    ##############################################################################
    def _init_agent(self):
        system_instructions = """You are an AI model serving as an academic advisor for the Open University of Israel (OUI). 
                                 The language of the OUI is Hebrew.
                                 You use tools to provide answers in a concise manner. 
                                 The name of the OUI in Hebrew is האוניברסיטה הפתוחה. 
                                 The tools give you access to a database of OUI courses.
                                 You can use these tools to answer questions about the courses.
                                 Each course has a unique id, a unique name, and several other details.
                                 Courses have a list of available semesters in which they are offered, for example '2025א' is the first semester of 2025, 
                                 '2025ב' is the second semester of 2025, and '2025ג' is the third semester of 2025.
                                 Each course has one of more classifications ("סיווגים"), which is typically the name of the department that offers the course, followed by the name of the faculty.
                                 """

        ##############################################################################
        @tool("GetCourseNameFromID")
        def get_course_name_from_id(course_id: str) -> str:
            """Get the course name from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """
            print(f"In Tool: Getting course name for {course_id}")
            return self.course_by_id[course_id]['course_name']
        
        ##############################################################################
        @tool("GetCourseCreditsFromID")
        def get_course_credits_from_id(course_id: str) -> str:
            """Get the course credits from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """
            print(f"In Tool: Getting course credits for {course_id}")
            return self.course_by_id[course_id]['credits']
        
        ##############################################################################
        @tool("GetCourseUrlFromID")
        def get_course_url_from_id(course_id: str) -> str:
            """Get the course url from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """
            print(f"In Tool: Getting course url for {course_id}")
            return self.course_by_id[course_id]['course_url']
        
        ##############################################################################
        @tool("GetCourseClassificationsFromID")
        def get_course_classifications_from_id(course_id: str) -> str:
            """Get the course classifications from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """
            print(f"In Tool: Getting course classifications for {course_id}")
            return self.course_by_id[course_id]['classification']
        
        ##############################################################################
        @tool("GetCourseDependenciesFromID")
        def get_course_dependencies_from_id(course_id: str) -> str:
            """Get the course dependencies from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """
            print(f"In Tool: Getting course dependencies for {course_id}")
            deps = []
            deps.append(self.course_by_id[course_id]['required_dependencies'])
            deps.append(self.course_by_id[course_id]['recommended_dependencies'])
            return deps
        
        ##############################################################################
        @tool("GetCourseSemestersFromID")
        def get_course_semesters_from_id(course_id: str) -> str:
            """Get the course semesters from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """
            print(f"In Tool: Getting course semesters for {course_id}")
            return self.course_by_id[course_id]['semesters']
        
        ##############################################################################
        tools = [
                    get_course_name_from_id, 
                    get_course_url_from_id, 
                    get_course_credits_from_id, 
                    get_course_classifications_from_id,
                    get_course_dependencies_from_id,
                    get_course_semesters_from_id
                ]

        # Pull the prompt template from the hub
        self.prompt_template = hub.pull("hwchase17/openai-tools-agent")

        # Create the ReAct agent using the create_tool_calling_agent function
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=tools,
            prompt=self.prompt_template
        )

        # Create the agent executor
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
        )

        self.agent = agent_executor

    ##############################################################################
    def init(self):
        # Initialize a ChatOpenAI model
        self.llm = ChatOpenAI(model=self.config["llm_model"])

        # Initialize the agent
        self._init_agent()

        # Initialize the data
        self._init_data()

    ##############################################################################
    def _format_chat_history(self, chat_history: list[dict]) -> list[dict]:
        formatted_history = []
        for entry in chat_history:
            formatted_history.append({
                "role": entry["sender"],
                "content": entry["message"]
            })
        return formatted_history 

    ##############################################################################
    def do_query(self, user_input: str, chat_history: list[dict]) -> str:
        formatted_chat_history = self._format_chat_history(chat_history)
        response = self.agent.invoke({"input": user_input, "chat_history": formatted_chat_history})
        print(f"Agent Response: {response}")
        return response['output']

##############################################################################
if __name__ == "__main__":
    from config import all_config

    llm = CoursesWithTools("Courses", all_config["General"])
    llm.init()

    # Test the agent with sample queries
    response = llm.do_query("What is the name of the course with ID 20905?", [])
    print(f"Agent Response: {response}")



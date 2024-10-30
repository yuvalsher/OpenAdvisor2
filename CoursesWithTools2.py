from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain import hub
import os
import json

# Global variables
faculty_code = None
config = None
course_data = []
course_by_id = {}
course_by_name = {}
llm = None
hub_prompt = None

def initialize(faculty_code_param, config_param):
    global faculty_code, config, course_data, course_by_id, course_by_name, llm, hub_prompt
    
    faculty_code = faculty_code_param
    config = config_param

    # Use the DB_Path from the config
    full_path = os.path.join(config["DB_Path"], "all_courses.json")

    # if the file does not exist - throw exception
    if not os.path.exists(full_path):
        print(f"File {full_path} does not exist.")
        raise FileNotFoundError(f"File {full_path} does not exist.")
    
    with open(full_path, "r", encoding='utf-8') as f:
        course_data = json.load(f)

    for course in course_data:
        course_by_id[course['course_id']] = course
        course_by_name[course['course_name']] = course

    # Initialize a ChatOpenAI model
    llm = ChatOpenAI(model=config["llm_model"])

    # Pull the prompt template from the hub
    hub_prompt = hub.pull("hwchase17/openai-tools-agent")

def _build_agent():
    # Define input schema for the tool
    class CourseIDSchema(BaseModel):
        course_id: str = Field(..., description="The ID of the course to look up")

    @tool("GetCourseNameFromID")
    def get_course_name_from_id(course_id: str) -> str:
        """Get the course name from the course ID.
        
        Args:
            course_id: The ID of the course to look up
        """
        print(f"In Tool: Getting course name for {course_id}")
        return course_by_id[course_id]['course_name']
    
    tools = [get_course_name_from_id]

    # Create the ReAct agent using the create_tool_calling_agent function
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=hub_prompt,
    )

    # Create the agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    return agent_executor

def do_query(user_input: str, chat_history: list[dict]) -> str:
    agent = _build_agent()
    response = agent.invoke({"input": user_input})
    print(f"Agent Response: {response}")
    return response

if __name__ == "__main__":
    # Test code
    from config import all_config
    initialize("Courses", all_config["General"])
    response = do_query("What is the name of the course with ID 20905?", [])
    print(f"Response: {response}") 
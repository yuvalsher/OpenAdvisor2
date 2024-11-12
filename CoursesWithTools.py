# Docs: https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/
# 
import os
import json
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain_core.tools import Tool, StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from AbstractLlm import AbstractLlm
from rag import Rag
from typing import Callable
#from toolz import curry


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
        self.memory = None
        self.system_instructions = """
            You are an AI model serving as an academic advisor for the Open University of Israel (OUI). 
            The language of the OUI is Hebrew.
            You use tools to provide answers in a concise manner. 
            The name of the OUI in Hebrew is האוניברסיטה הפתוחה. 
            The tools give you access to a database of OUI courses. You can use these tools to answer questions about the courses.
            Each course has a unique id (or course number), a unique name, and several other details. 
            There are specific tools for getting most of the course important details, and also a tool to get the course overview, which contains all these details and many more.
            == Semesters ==
            Courses have a list of available semesters in which they are offered, for example '2025א' is the first semester of 2025, 
            '2025ב' is the second semester of 2025, and '2025ג' is the third semester (summer semester) of 2025.
            == Classifications ==
            Each course has one of more classifications ("סיווגים"), which are typically the name of the department that offers the course, 
            followed by the name of the faculty.
            == Overlaps ==
            Some courses have overlaps with other courses. The list of courses that have full or partial overlap with a given course is given by the 'overlap_courses' tool.
            In case of an overlap, the details of the overlapping courses are given by the 'overlap_url' tool.
            If the user asks specifically about overlaps, the overlap url should be given.
            == Dependencies ==
            Some courses have dependencies, which are courses that must be taken before the current course. This is called "תנאי קבלה" or "ידע קודם" or "קורסי קדם".
            There are three types of dependencies: required dependencies, recommended dependencies, and condition dependencies.
            Condition dependencies ("תנאי קבלה") are conditions (such as courses that must be taken before the current course), without which registration in the current course will fail.
            Required dependencies ("ידע קודם דרוש") are courses that should be taken before the current course.
            Recommended dependencies ("ידע קודם מומלץ") are courses that are recommended to be taken before the current course.
            Each type of dependency has two tools: one that returns the text of the dependency, and one that returns the list of courses that are dependencies.
            Sometimes dependent courses have overlaps, so only a subset of them must be taken.
            == Similarity check ==
            The tool GetSimilarCourses uses an embeddings vector search to find courses that are similar to the search text. 
            It returns a fixed number of results, so some of them may not be relevant. Each result must be checked for its details to verify it is indeed relevant to the query.
            """


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

        ##############################################################################
        def create_tool_1(course_id: str, info: str) -> str:
            """Get course information from the course ID.
         
            Args:
                course_id: The ID of the course to look up
                info: the label of the requested information, such as: 'course_name', 'course_url', 'condition_url', etc.
            """
            
            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found")
                return None
                
            result = self.course_by_id[course_id][info]
            print(f"not sure this works {course_id} {info}: {result[::-1]}")
            
            return result
            
        #@curry
        def create_tool_2(course_id: str, info: str) -> str:
            """Get the course classifications from the course ID.
         
            Args:
                course_id: The ID of the course to look up
            """
            
            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found")
                return None
                
            def get_info() -> str:
                result = self.course_by_id[course_id][info]
                print(f"not sure this works {course_id} {info}: {result[::-1]}")
                return result
                
            return get_info
        
        def create_tool_3(course_id: str) -> Callable[[str], str]:
            """Get the course classifications from the course ID.
         
            Args:
                course_id: The ID of the course to look up
            """
            
            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found")
                return None
                
            def get_info(info: str) -> str:
                result = self.course_by_id[course_id][info]
                print(f"not sure this works {course_id} {info}: {result[::-1]}")
                return result
                
            return get_info

        ##############################################################################
        @tool("GetCourseIDFromName")
        def get_course_id_from_name(course_name: str) -> str:
            """Get the course ID from the course name.
            
            Args:
                course_name: The name of the course to look up
            """

            if course_name not in self.course_by_name:
                print(f"In Tool: Course {course_name[::-1]} not found")
                return None
            
            result = self.course_by_name[course_name]['course_id']
            print(f"In Tool: Getting course ID for {course_name[::-1]}: {result}\n")
            return result
        
        ##############################################################################
        @tool("GetCourseNameFromID")
        def get_course_name_from_id(course_id: str) -> str:
            """Get the course name from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """

            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found")
                return None
            
            result = self.course_by_id[course_id]['course_name']
            print(f"In Tool: Getting course name for {course_id}: {result[::-1]}\n")
            return result
        
        ##############################################################################
        @tool("GetCourseCreditsFromID")
        def get_course_credits_from_id(course_id: str) -> str:
            """Get the course credits from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """

            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found")
                return None
            
            result = self.course_by_id[course_id]['credits']
            print(f"In Tool: Getting course credits for {course_id}: {result[::-1]}\n")
            return result
        
        ##############################################################################
        @tool("GetCourseUrlFromID")
        def get_course_url_from_id(course_id: str) -> str:
            """Get the course url from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """

            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found")
                return None
            
            result = self.course_by_id[course_id]['course_url']
            print(f"In Tool: Getting course url for {course_id}: {result}\n")
            return result
        
        ##############################################################################
        @tool("GetCourseClassificationsFromID")
        def get_course_classifications_from_id(course_id: str) -> str:
            """Get the course classifications from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """

            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found")
                return None
            
            result = self.course_by_id[course_id]['classification']
            print(f"In Tool: Getting course classifications for {course_id}: {result[::-1]}")
            return result

        ##############################################################################
        @tool("GetCourseConditionDependenciesTextFromID")
        def get_course_condition_dependencies_text_from_id(course_id: str) -> str:
            """Get the course condition dependencies text from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """

            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found\n")
                return None
            
            result = self.course_by_id[course_id]['condition_text']
            print(f"In Tool: Getting condition dependencies text for {course_id}: {result}\n")
            return result

        ##############################################################################
        @tool("GetCourseConditionDependenciesCoursesFromID")
        def get_course_condition_dependencies_courses_from_id(course_id: str) -> str:
            """Get the course condition dependencies courses from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """
            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found\n")
                return None
            
            result = self.course_by_id[course_id]['condition_courses']
            print(f"In Tool: Getting condition dependencies courses for {course_id}: {result}\n")
            return result

        ##############################################################################
        @tool("GetCourseRequiredDependenciesTextFromID")
        def get_course_required_dependencies_text_from_id(course_id: str) -> str:
            """Get the course required dependencies text from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """
            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found\n")
                return None
            
            result = self.course_by_id[course_id]['required_deps_text']
            print(f"In Tool: Getting required dependencies text for {course_id}: {result[::-1]}\n")
            return result

        ##############################################################################
        @tool("GetCourseRequiredDependenciesCoursesFromID")
        def get_course_required_dependencies_courses_from_id(course_id: str) -> str:
            """Get the course required dependencies courses from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """
            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found\n")
                return None
            
            result = self.course_by_id[course_id]['required_deps_courses']
            print(f"In Tool: Getting required dependencies courses for {course_id}: {result}\n")
            return result

        ##############################################################################
        @tool("GetCourseRecommendedDependenciesTextFromID")
        def get_course_recommended_dependencies_text_from_id(course_id: str) -> str:
            """Get the course recommended dependencies text from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """
            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found\n")
                return None
            
            result = self.course_by_id[course_id]['recommended_deps_text']
            print(f"In Tool: Getting recommended dependencies text for {course_id}: {result[::-1]}\n")
            return result

        ##############################################################################
        @tool("GetCourseRecommendedDependenciesCoursesFromID")
        def get_course_recommended_dependencies_courses_from_id(course_id: str) -> str:
            """Get the course recommended dependencies courses from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """
            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found\n")
                return None
            
            result = self.course_by_id[course_id]['recommended_deps_courses']
            print(f"In Tool: Getting recommended dependencies courses for {course_id}: {result}\n")
            return result

        ##############################################################################
        @tool("GetAllDependenciesCoursesFromID")
        def get_all_dependencies_courses_from_id(course_id: str) -> str:
            """Get the all the dependencies courses (condition, required and recommended) from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """
            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found\n")
                return None
            
            deps = []   
            deps.extend(self.course_by_id[course_id]['condition_courses'])
            deps.extend(self.course_by_id[course_id]['required_deps_courses'])
            deps.extend(self.course_by_id[course_id]['recommended_deps_courses'])
            print(f"In Tool: Getting all dependencies courses for {course_id}: {deps}\n")
            return deps

        ##############################################################################
        @tool("GetCourseSemestersFromID")
        def get_course_semesters_from_id(course_id: str) -> str:
            """Get the course semesters from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """

            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found\n")
                return None
            
            result = self.course_by_id[course_id]['semesters']
            print(f"In Tool: Getting course semesters for {course_id}: {result}\n")
            return result
        
        ##############################################################################
        @tool("GetCourseOverviewFromID")
        def get_course_overview_from_id(course_id: str) -> str:
            """Get the course overview from the course ID. 
            
            Args:
                course_id: The ID of the course to look up
            """

            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found\n")
                return None
            
            all_text = self.course_by_id[course_id]['text']
            # concatenate all_text from an array of strings to a single string
            result = ' '.join(all_text)
            print(f"In Tool: Getting course overview for {course_id}: {result[::-1]}\n")
            return result
        
        ##############################################################################
        @tool("GetCourseOverlapUrlFromID")
        def get_course_overlap_url_from_id(course_id: str) -> str:
            """Get the course overlap url from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """

            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found\n")
                return None
            
            if 'overlap_url' not in self.course_by_id[course_id]:
                return None
            
            result = self.course_by_id[course_id]['overlap_url']
            print(f"In Tool: Getting course overlap url for {course_id}: {result}\n")
            return result
        
        ##############################################################################
        @tool("GetCourseOverlapCoursesFromID")
        def get_course_overlap_courses_from_id(course_id: str) -> str:
            """Get the course overlap courses from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """

            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found\n")
                return None
            
            if 'overlap_courses' not in self.course_by_id[course_id]:
                return None
            
            result = self.course_by_id[course_id]['overlap_courses']
            print(f"In Tool: Getting course overlap courses for {course_id}: {result}\n")
            return result
        
        ##############################################################################
        @tool("GetSimilarCoursesByText")
        def get_similar_courses_by_text(query_text: str) -> str:
            """Find courses with similar content to the query text by using an embeddings vector search.
.            
            Args:
                query_text: A subject or topic to search for in the course overview
            """

            result = self.rag.retrieve_rag_chunks_for_tool(query_text)
            print(f"In Tool: Getting similar courses for {query_text}: {result}\n")
            return result
            
       ##############################################################################
        @tool("GetSimilarCoursesByID")
        def get_similar_courses_by_id(course_id: str) -> str:
            """Find courses with similar content to the overview of the input course, by using an embeddings vector search.
.            
            Args:
                course_id: The course ID of the input course
            """

            overview = get_course_overview_from_id(course_id)
            result = self.rag.retrieve_rag_chunks_for_tool(overview)
            print(f"In Tool: Getting similar courses for {course_id}: {result}\n")
            return result
            
        ##############################################################################
        tools = [
                    get_course_id_from_name, 
                    #create_tool_1,
                    get_course_name_from_id, 
                    get_course_url_from_id, 
                    get_course_credits_from_id, 
                    get_course_classifications_from_id,
                    get_course_condition_dependencies_text_from_id,
                    get_course_condition_dependencies_courses_from_id,
                    get_course_required_dependencies_text_from_id,
                    get_course_required_dependencies_courses_from_id,
                    get_course_recommended_dependencies_text_from_id,
                    get_course_recommended_dependencies_courses_from_id,
                    get_all_dependencies_courses_from_id,
                    get_course_semesters_from_id,
                    get_course_overview_from_id,
                    get_course_overlap_url_from_id,
                    get_course_overlap_courses_from_id,
                    get_similar_courses_by_id,
                    get_similar_courses_by_text,
                ]

        # Pull the prompt template from the hub
        self.prompt_template = hub.pull("hwchase17/openai-tools-agent")

        # ConversationBufferMemory stores the conversation history, allowing the agent to maintain context across interactions
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )

        self.memory.chat_memory.add_message(SystemMessage(content=self.system_instructions))

        # Create the ReAct agent using the create_tool_calling_agent function
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=tools,
            prompt=self.prompt_template,
        )

        # Create the agent executor
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
        )

        self.agent = agent_executor

    ##############################################################################
    def init(self):
        # Initialize a ChatOpenAI model
        self.llm = ChatOpenAI(model=self.config["llm_model"])

        self.rag = Rag(self.faculty_code, self.config)
        self.rag.init()

        # Initialize the agent
        self._init_agent()

        # Initialize the data
        self._init_data()

    ##############################################################################
    """
    def _format_chat_history(self, chat_history: list[dict]) -> list[dict]:
        formatted_history = []
        for entry in chat_history:
            formatted_history.append({
                "role": entry["sender"],
                "content": entry["message"]
            })
        return formatted_history 
    """

    ##############################################################################
    def do_query(self, user_input: str, chat_history: list[dict]) -> str:
        # Ignore Dash chat history - use memory instead.
        #formatted_chat_history = self._format_chat_history(chat_history)
        #response = self.agent.invoke({"input": user_input, "chat_history": formatted_chat_history})

        # Add the user's message to the conversation memory - agent.invoke() already does that
        #self.memory.chat_memory.add_message(HumanMessage(content=user_input))

        response = self.agent.invoke({"input": user_input})
        print(f"Agent Response: {response}")

        # Add the agent's response to the conversation memory - agent.invoke() already does that
        #self.memory.chat_memory.add_message(AIMessage(content=response["output"]))

        return response['output']

    ##############################################################################
    def reset_chat_history(self):
        self.memory.clear()
        self.memory.chat_memory.add_message(SystemMessage(content=self.system_instructions))


##############################################################################
if __name__ == "__main__":
    from OpenAdvisor2 import main
    main("Tools", "CS")
    
    # from config import all_config

    # llm = CoursesWithTools("Courses", all_config["General"])
    # llm.init()

    # # Test the agent with sample queries
    # response = llm.do_query("What is the name of the course with ID 20905?", [])
    # print(f"Agent Response: {response}")



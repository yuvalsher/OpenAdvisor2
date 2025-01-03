# Docs: https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/
# 
import os
import json
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent, initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain_core.tools import Tool, StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from AbstractLlm import AbstractLlm
from rag import Rag
from typing import Callable, Dict, Type, List, Optional
from uuid import uuid4


##############################################################################
###### CoursesWithTools class  
##############################################################################
class CoursesWithTools(AbstractLlm):

    ##############################################################################
    def __init__(self, config):
        super().__init__(config)
        self.course_data = []
        self.course_by_id = {}
        self.course_by_name = {}
        self.memories: Dict[str, ConversationBufferMemory] = {}
        
        self.system_instructions = """
            You are an advanced AI assistant with vast knowledge of the Open University of Israel (האוניברסיטה הפתוחה), designed to provide helpful information.  
            Your primary function is to assist users by answering questions, offering explanations, and providing insights, adapting your communication style to suit different users and contexts. 
            The language of most of the content is Hebrew. Hebrew names for elements such as course names, faculty names, study program names, etc. must be provided verbatim, exactly as given in the provided content (returned by the provided tools).
            
            **Role and Tools:**
            - Utilize various tools to provide concise and accurate answers.
            - Access a comprehensive database of OUI courses through these tools to respond to user inquiries about course offerings and details.

            **Course Structure:**
            - Each course has a unique ID (course number), a unique name, and additional pertinent details.
            - Utilize specific tools to retrieve essential course information and an overview that encompasses all relevant details.

            **Semesters:**
            - Courses are offered in specific semesters:
                - '2025א' – First semester of 2025
                - '2025ב' – Second semester of 2025
                - '2025ג' – Summer semester of 2025

            **Classifications:**
            - Courses are classified under one or more departments ("סיווגים"), typically indicating the department and faculty offering the course.

            **Overlaps:**
            - Some courses overlap with others. Use the 'overlap_courses' tool to list courses with full or partial overlaps.
            - For detailed information on overlapping courses, utilize the 'overlap_url' tool.
            - If a user inquires specifically about overlaps, provide the overlap URL.

            **Dependencies:**
            - Courses may have dependencies that must be satisfied before enrollment:
                - **Condition Dependencies ("תנאי קבלה")**: Mandatory prerequisites without which registration fails.
                - **Required Dependencies ("ידע קודם דרוש")**: Courses that should be taken prior.
                - **Recommended Dependencies ("ידע קודם מומלץ")**: Courses that are advised to be taken beforehand.
            - Each dependency type has two associated tools:
                - One returns the textual description of the dependencies.
                - The other returns a list of courses that are dependencies.
            - Note that some dependent courses may have overlaps, requiring only a subset to be completed.

            **Similarity Check:**
            - The tool `GetSimilarCourses` employs an embeddings vector search to identify courses similar to the input query text.
            - It returns a fixed number of results which may include irrelevant courses. Each result should be reviewed to ensure relevance.

            **General Guidelines:**
            - Ensure all responses are clear, concise, and relevant.
            - Leverage the available tools effectively to provide accurate information.
            - Maintain the flow of conversation and context using the conversation memory.
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
    def _init_tools(self):

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
        class CourseDetails(BaseModel):
            course_url: str
            course_id: str
            course_name: str
            credits: Optional[str] = Field(default="")
            classification: List[List[str]]
            condition_text: Optional[str] = Field(default="")
            condition_courses: List[str]
            required_deps_text: Optional[str] = Field(default="")
            required_deps_courses: List[str]
            recommended_deps_text: Optional[str] = Field(default="")
            recommended_deps_courses: List[str]
            overlap: List[str]
            text: List[str]
            footnotes: List[str]
            discontinued: bool
            overlap_url: str
            semesters: List[str]
            overlap_courses: List[str]

        @tool("GetCourseDetailsFromID")
        def get_course_details_from_id(course_id: str) -> CourseDetails:
            """Get the course details from the course ID.
            
            Args:
                course_id: The ID of the course to look up
            """

            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found")
                return None
            
            result = self.course_by_id[course_id]
            print(f"In Tool: Getting course details for {course_id}: {result['course_name'][::-1]}\n")
            return result
        
        # ##############################################################################
        # @tool("GetCourseCreditsFromID")
        # def get_course_credits_from_id(course_id: str) -> str:
        #     """Get the course credits from the course ID.
            
        #     Args:
        #         course_id: The ID of the course to look up
        #     """

        #     if course_id not in self.course_by_id:
        #         print(f"In Tool: Course {course_id} not found")
        #         return None
            
        #     result = self.course_by_id[course_id]['credits']
        #     print(f"In Tool: Getting course credits for {course_id}: {result[::-1]}\n")
        #     return result
        
        # ##############################################################################
        # @tool("GetCourseUrlFromID")
        # def get_course_url_from_id(course_id: str) -> str:
        #     """Get the course url from the course ID.
            
        #     Args:
        #         course_id: The ID of the course to look up
        #     """

        #     if course_id not in self.course_by_id:
        #         print(f"In Tool: Course {course_id} not found")
        #         return None
            
        #     result = self.course_by_id[course_id]['course_url']
        #     print(f"In Tool: Getting course url for {course_id}: {result}\n")
        #     return result
        
        # ##############################################################################
        # @tool("GetCourseClassificationsFromID")
        # def get_course_classifications_from_id(course_id: str) -> str:
        #     """Get the course classifications from the course ID.
            
        #     Args:
        #         course_id: The ID of the course to look up
        #     """

        #     if course_id not in self.course_by_id:
        #         print(f"In Tool: Course {course_id} not found")
        #         return None
            
        #     result = self.course_by_id[course_id]['classification']
        #     print(f"In Tool: Getting course classifications for {course_id}: {result[::-1]}")
        #     return result

        # ##############################################################################
        # @tool("GetCourseConditionDependenciesTextFromID")
        # def get_course_condition_dependencies_text_from_id(course_id: str) -> str:
        #     """Get the course condition dependencies text from the course ID.
            
        #     Args:
        #         course_id: The ID of the course to look up
        #     """

        #     if course_id not in self.course_by_id:
        #         print(f"In Tool: Course {course_id} not found\n")
        #         return None
            
        #     result = self.course_by_id[course_id]['condition_text']
        #     print(f"In Tool: Getting condition dependencies text for {course_id}: {result}\n")
        #     return result

        # ##############################################################################
        # @tool("GetCourseConditionDependenciesCoursesFromID")
        # def get_course_condition_dependencies_courses_from_id(course_id: str) -> str:
        #     """Get the course condition dependencies courses from the course ID.
            
        #     Args:
        #         course_id: The ID of the course to look up
        #     """
        #     if course_id not in self.course_by_id:
        #         print(f"In Tool: Course {course_id} not found\n")
        #         return None
            
        #     result = self.course_by_id[course_id]['condition_courses']
        #     print(f"In Tool: Getting condition dependencies courses for {course_id}: {result}\n")
        #     return result

        # ##############################################################################
        # @tool("GetCourseRequiredDependenciesTextFromID")
        # def get_course_required_dependencies_text_from_id(course_id: str) -> str:
        #     """Get the course required dependencies text from the course ID.
            
        #     Args:
        #         course_id: The ID of the course to look up
        #     """
        #     if course_id not in self.course_by_id:
        #         print(f"In Tool: Course {course_id} not found\n")
        #         return None
            
        #     result = self.course_by_id[course_id]['required_deps_text']
        #     print(f"In Tool: Getting required dependencies text for {course_id}: {result[::-1]}\n")
        #     return result

        # ##############################################################################
        # @tool("GetCourseRequiredDependenciesCoursesFromID")
        # def get_course_required_dependencies_courses_from_id(course_id: str) -> str:
        #     """Get the course required dependencies courses from the course ID.
            
        #     Args:
        #         course_id: The ID of the course to look up
        #     """
        #     if course_id not in self.course_by_id:
        #         print(f"In Tool: Course {course_id} not found\n")
        #         return None
            
        #     result = self.course_by_id[course_id]['required_deps_courses']
        #     print(f"In Tool: Getting required dependencies courses for {course_id}: {result}\n")
        #     return result

        # ##############################################################################
        # @tool("GetCourseRecommendedDependenciesTextFromID")
        # def get_course_recommended_dependencies_text_from_id(course_id: str) -> str:
        #     """Get the course recommended dependencies text from the course ID.
            
        #     Args:
        #         course_id: The ID of the course to look up
        #     """
        #     if course_id not in self.course_by_id:
        #         print(f"In Tool: Course {course_id} not found\n")
        #         return None
            
        #     result = self.course_by_id[course_id]['recommended_deps_text']
        #     print(f"In Tool: Getting recommended dependencies text for {course_id}: {result[::-1]}\n")
        #     return result

        # ##############################################################################
        # @tool("GetCourseRecommendedDependenciesCoursesFromID")
        # def get_course_recommended_dependencies_courses_from_id(course_id: str) -> str:
        #     """Get the course recommended dependencies courses from the course ID.
            
        #     Args:
        #         course_id: The ID of the course to look up
        #     """
        #     if course_id not in self.course_by_id:
        #         print(f"In Tool: Course {course_id} not found\n")
        #         return None
            
        #     result = self.course_by_id[course_id]['recommended_deps_courses']
        #     print(f"In Tool: Getting recommended dependencies courses for {course_id}: {result}\n")
        #     return result

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

        # ##############################################################################
        # @tool("GetCourseSemestersFromID")
        # def get_course_semesters_from_id(course_id: str) -> str:
        #     """Get the course semesters from the course ID.
            
        #     Args:
        #         course_id: The ID of the course to look up
        #     """

        #     if course_id not in self.course_by_id:
        #         print(f"In Tool: Course {course_id} not found\n")
        #         return None
            
        #     result = self.course_by_id[course_id]['semesters']
        #     print(f"In Tool: Getting course semesters for {course_id}: {result}\n")
        #     return result
        
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
        
        # ##############################################################################
        # @tool("GetCourseOverlapUrlFromID")
        # def get_course_overlap_url_from_id(course_id: str) -> str:
        #     """Get the course overlap url from the course ID.
            
        #     Args:
        #         course_id: The ID of the course to look up
        #     """

        #     if course_id not in self.course_by_id:
        #         print(f"In Tool: Course {course_id} not found\n")
        #         return None
            
        #     if 'overlap_url' not in self.course_by_id[course_id]:
        #         return None
            
        #     result = self.course_by_id[course_id]['overlap_url']
        #     print(f"In Tool: Getting course overlap url for {course_id}: {result}\n")
        #     return result
        
        # ##############################################################################
        # @tool("GetCourseOverlapCoursesFromID")
        # def get_course_overlap_courses_from_id(course_id: str) -> str:
        #     """Get the course overlap courses from the course ID.
            
        #     Args:
        #         course_id: The ID of the course to look up
        #     """

        #     if course_id not in self.course_by_id:
        #         print(f"In Tool: Course {course_id} not found\n")
        #         return None
            
        #     if 'overlap_courses' not in self.course_by_id[course_id]:
        #         return None
            
        #     result = self.course_by_id[course_id]['overlap_courses']
        #     print(f"In Tool: Getting course overlap courses for {course_id}: {result}\n")
        #     return result
        
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
        self.tools = [
                    get_course_id_from_name, 
                    get_course_name_from_id, 
                    get_course_details_from_id,
                    # get_course_url_from_id, 
                    # get_course_credits_from_id, 
                    # get_course_classifications_from_id,
                    # get_course_condition_dependencies_text_from_id,
                    # get_course_condition_dependencies_courses_from_id,
                    # get_course_required_dependencies_text_from_id,
                    # get_course_required_dependencies_courses_from_id,
                    # get_course_recommended_dependencies_text_from_id,
                    # get_course_recommended_dependencies_courses_from_id,
                    get_all_dependencies_courses_from_id,
                    # get_course_semesters_from_id,
                    # get_course_overview_from_id,
                    # get_course_overlap_url_from_id,
                    # get_course_overlap_courses_from_id,
                    get_similar_courses_by_id,
                    get_similar_courses_by_text,
                ]

    ##############################################################################
    def get_agent(self, client_id: str):

        # Pull the prompt template from the hub
        self.prompt_template = hub.pull("hwchase17/openai-tools-agent")

        memory = self._get_or_create_memory(client_id)

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
            memory=memory,
            handle_parsing_errors=True,
        )

        return agent_executor
    
    ##############################################################################
    def init(self, faculty_code):
        self.faculty_code = faculty_code
        # Initialize a ChatOpenAI model
        self.llm = ChatOpenAI(model=self.config["llm_model"])

        self.rag = Rag(self.config)
        self.rag.init(self.faculty_code)

        # Initialize the data
        self._init_data()

        self._init_tools()

    ##############################################################################
    def do_query(self, user_input: str, chat_history: list[dict], client_id: str = None) -> tuple[str, str]:
        """
        Process a query from a client.
        
        Args:
            user_input: The user's query
            chat_history: The chat history (ignored as we use memory)
            client_id: The client's unique identifier
            
        Returns:
            tuple: (response_text, client_id)
        """
        agent = self.get_agent(client_id)
        response = agent.invoke({"input": user_input})
        print(f"Agent Response for client {client_id}: {response}")
        return response['output'], client_id

    ##############################################################################
    def reset_chat_history(self, client_id: str):
        """Reset chat history for a specific client."""
        if client_id in self.memories:
            self.memories[client_id].clear()
            self.memories[client_id].chat_memory.add_message(
                SystemMessage(content=self.system_instructions)
            )


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



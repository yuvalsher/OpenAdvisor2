import os
import json
from langchain.agents import tool
from pydantic import BaseModel, Field
from typing import Optional, List
import PyPDF2
from streamlit.runtime.uploaded_file_manager import UploadedFile
import io  # Also need to import io for BytesIO

from AbstractAgent import AbstractAgent
from rag import Rag
from OpenAI_Assistant import OpenAIAssistant
#from StudyProgramAgent import StudyProgramAgent
from utils import flip_by_line

class RouterAgent(AbstractAgent):

    ##############################################################################
    def __init__(self, config):
        super().__init__(config)
        self.course_data = []
        self.course_by_id = {}
        self.course_by_name = {}
        self.study_programs = {}
        
    ##############################################################################
    def _init_data(self):
        self.system_instructions = """
            You are an advanced AI assistant with vast knowledge of the Open University of Israel (האוניברסיטה הפתוחה), designed to provide helpful information.  
            Your primary function is to assist users by answering questions, offering explanations, and providing insights, adapting your communication style to suit different users and contexts. 
            The language of most of the content is Hebrew. Hebrew names for elements such as course names, faculty names, study program names, etc. must be provided verbatim, exactly as given in the provided content (returned by the provided tools).
            The query may include the text contents of PDF files uploaded by the user. Use this content to answer the user query.
            
            - **Your Task:**
                Your task is to analyze the user query and decide how it should be handled. 
                First identify if the question involves a specific study program or not. If it does, act on the first category below. If it doesn't, act on the second or thirdcategory.
                The query will fall into one of the following categories:

                - 1. Study Programs: Questions about specific study programs offered by the university. Study programs are a collection of requirements for eligibility for an academic degree. Study program details involve several sections, each of them can be a list of elective or required courses, with a requirement for minimum credit points. These queries should be handled using the a study program tool. The study program tool requires the study program code as input. If a study program name is provided in the user query, match the name to the list of study program names and codes. If the study program name is not provided in the query, you must ask the user for the study program name to identify the code. If you are unsure about the study program name, you can ask the user to approve your choice, or provide the study program name. Once you have the study program code, use the study program tool with the study program code and the question that the tool should answer. The tool does not have access to the chat history, so you must rephrase the question so that it contains all the relevant details from the chat history. If the query includes the contents of a PDF file, with the grade status of the student, provide that content to the study program tool as a parameter. 
                - 2. Course Details: Questions about specific university courses. Use the courses tools to answer these questions. Course details include course ID, name, URL, credits, classification, dependent courses, course overlaps, course overview, and available semesters.
                - 3. General University Information: General questions about studying at the university. Use the GetRelevantContent tool to search for relevant information from the university website. Always provide the source url of the information in the response.
             
            - **Expected Response Format:**
                If the user query is unclear or missing necessary information, such as the name of the study program, the response should be a clarifying question for gathering the required details from the user.
                Otherwise the response should be a detailed answer to the user question.

            - **Role and Tools:**
                - Utilize various tools to provide concise and accurate answers.
                - Access a comprehensive database of OUI courses through these tools to respond to user inquiries about course offerings and details.

            - **General Guidelines:**
                - Ensure all responses are clear, concise, and relevant.
                - Leverage the available tools effectively to provide accurate information.
                - Maintain the flow of conversation and context using the conversation memory.

            - **Course Structure:**
                Each course has a unique ID (course number), a unique name, and additional pertinent details.
                Utilize specific tools to retrieve essential course information and an overview that encompasses all relevant details.

                - Semesters:
                    - Courses are offered in specific semesters:
                        - '2025א' – First (winter) semester of 2025
                        - '2025ב' – Second (spring) semester of 2025
                        - '2025ג' – Third (summer) semester of 2025

                - Classifications:
                    - Courses are classified under one or more departments ("סיווגים"), typically indicating the department and faculty offering the course.

                - Overlaps:
                    - Some courses overlap with others. Use the 'overlap_courses' tool to list courses with full or partial overlaps.
                    - For detailed information on overlapping courses, utilize the 'overlap_url' tool.
                    - If a user inquires specifically about overlaps, provide the overlap URL.

                - Dependencies:
                    - Courses may have dependencies that must be satisfied before enrollment:
                        - **Condition Dependencies ("תנאי קבלה")**: Mandatory prerequisites without which registration fails.
                        - **Required Dependencies ("ידע קודם דרוש")**: Courses that should be taken prior.
                    - **Recommended Dependencies ("ידע קודם מומלץ")**: Courses that are advised to be taken beforehand.
                    - Each dependency type has two associated tools:
                        - One returns the textual description of the dependencies.
                        - The other returns a list of courses that are dependencies.
                    - Note that some dependent courses may have overlaps, requiring only a subset to be completed.

                - Similarity Check:
                    - The tool `GetSimilarCourses` employs an embeddings vector search to identify courses similar to the input query text.
                    - It returns a fixed number of results which may include irrelevant courses. Each result should be reviewed to ensure relevance.
            - **IMPORTANT:**
                - If the query includes the contents of a PDF file, with the grade status of the student, provide that content to the study program tool as a parameter.
            """

        self.prompt = """
            ענה על שאלת הסטודנט הבאה:\n
        """

        self.course_data = self._load_json_file("all_courses.json")
        for course in self.course_data:
            self.course_by_id[course['course_id']] = course
            self.course_by_name[course['course_name']] = course

        self.program_data = self._load_json_file("cs_study_programs.json")
        for program in self.program_data:
            self.study_programs[program['name']] = program['code']

        self.courses_rag = Rag(self.config)
        self.courses_rag.init("Courses")

        self.website_rag = Rag(self.config)
        self.website_rag.init("All")

        # Create agent for OpenAI Assistant for Study Programs
        self.study_programs_assistant = OpenAIAssistant(self.config)
        self.study_programs_assistant.init()

        # self.study_program_agent = StudyProgramAgent(self.config)
        # self.study_program_agent.init()

    ##############################################################################
    def get_system_instructions(self):
        return [self.system_instructions]

    ##############################################################################
    def _init_tools(self):

        ##############################################################################
        @tool("GetCourseIDFromName")
        def _get_course_id_from_name(course_name: str) -> str:
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
        def _get_course_name_from_id(course_id: str) -> str:
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
        def _get_course_details_from_id(course_id: str) -> CourseDetails:
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
        
        ##############################################################################
        @tool("GetAllDependenciesCoursesFromID")
        def _get_all_dependencies_courses_from_id(course_id: str) -> List[str]:
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
        @tool("GetCourseOverviewFromID")
        def _get_course_overview_from_id(course_id: str) -> str:
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
        @tool("GetSimilarCoursesByText")
        def _get_similar_courses_by_text(query_text: str) -> str:
            """Find courses with similar content to the query text by using an embeddings vector search.
.            
            Args:
                query_text: A subject or topic to search for in the course overview
            """

            result = self.courses_rag.retrieve_rag_chunks_for_tool(query_text)
            print(f"In Tool: Getting similar courses for {query_text[::-1]}: {result[::-1]}\n")
            return result
            
       ##############################################################################
        @tool("GetSimilarCoursesByID")
        def _get_similar_courses_by_id(course_id: str) -> str:
            """Find courses with similar content to the overview of the input course, by using an embeddings vector search.
.            
            Args:
                course_id: The course ID of the input course
            """

            if course_id not in self.course_by_id:
                print(f"In Tool: Course {course_id} not found\n")
                return None
            
            all_text = self.course_by_id[course_id]['text']
            # concatenate all_text from an array of strings to a single string
            overview = '\n'.join(all_text)
            result = self.courses_rag.retrieve_rag_chunks_for_tool(overview)
            print(f"In Tool: Getting similar courses for {course_id[::-1]}: {result[::-1]}\n")
            return result
            
        ##############################################################################
        @tool("GetRelevantContent")
        def _get_relevant_content(query_text: str) -> str:
            """Find relevant content to the query text by using an embeddings vector search.
.            
            Args:
                query_text: A subject or topic to search for in the course overview
            """

            result = self.website_rag.retrieve_rag_chunks_for_tool(query_text)
            print(f"In Tool: Getting relevant content for query {query_text[::-1]}: \n{flip_by_line(result)}\n")
            return result
            
        ##############################################################################
        @tool("GetStudyProgramCodeFromName")
        def _get_study_program_code_from_name(study_program_name: str) -> str:
            """Get the study program code from the study program name.
.            
            Args:
                study_program_name: The name of the study program to look up
            """

            if study_program_name not in self.study_programs:
                print(f"In Tool: Study program {study_program_name} not found\n")
                return None
            
            result = self.study_programs[study_program_name]
            print(f"In Tool: Getting study program code for {study_program_name[::-1]}: {result}\n")
            return result
            
        ##############################################################################
        @tool("GetListOfStudyProgramNamesAndCodes")
        def _get_list_of_study_program_names_and_codes() -> List[str]:
            """Get a list of all study program names.

            Returns:
                List[str]: A list of study program names.
            """

            #result = [(code, name) for name, code in self.study_programs.items()]
            result = [name for name in self.study_programs.keys()]
            reversed_result = [name[::-1] for name in result]
            print(f"In Tool: Getting list of study program names: {reversed_result}\n")
            return result
        
        ##############################################################################
        @tool("GetAnswerOnStudyPrograms")
        def _get_answer_on_study_programs(query_text: str, study_program_code: str, grade_status: str) -> str:
            """Get an answer on study programs from the study programs assistant.
.            
            Args:
                query_text: A question about a study program
                study_program_code: The code of the study program to answer the question
                grade_status: A list of completed courses with their grades
            """
            # try:
            #     agent = self.study_program_agent.get_agent()
            #     memory = self.study_program_agent.create_new_memory()
            #     agent.memory = memory
            #     prompt = self.study_program_agent.get_prompt(study_program_code, query_text)
            #     response = agent.invoke({"input": prompt})
            #     result = response['output']
            # except Exception as e:
            #     print(f"Error in study program agent: {e}")
            #     result = "Error in study program agent"

            print(f"In Tool: Getting answer on study program {study_program_code} for {query_text[::-1]}\nIncluded grade status: {grade_status}")

            result = self.study_programs_assistant.do_query(query_text, study_program_code, grade_status)

            print(f"In Tool: Getting answer on study program {study_program_code} Returning: {result[::-1]}\n")
            return result
        
        ##############################################################################
        self.tools = [
                    _get_course_id_from_name, 
                    _get_course_name_from_id, 
                    _get_course_details_from_id,
                    #_get_all_dependencies_courses_from_id,
                    _get_similar_courses_by_text,
                    _get_similar_courses_by_id,
                    _get_relevant_content,
                    _get_study_program_code_from_name,
                    _get_list_of_study_program_names_and_codes,
                    _get_answer_on_study_programs,
                ]

    ##############################################################################
    def get_prompt(self):
        return self.prompt

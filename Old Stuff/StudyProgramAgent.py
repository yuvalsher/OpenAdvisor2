import os
import json
from langchain.agents import tool
from pydantic import BaseModel, Field
from typing import Optional, List

from AbstractAgent import AbstractAgent

class StudyProgramAgent(AbstractAgent):

    ##############################################################################
    def __init__(self, config):
        super().__init__(config)
        self.study_programs = {}

    ##############################################################################
    def _init_data(self):
        self.system_instructions = """
            You are an academic advisor for the Open University of Israel (האוניברסיטה הפתוחה), specializing in academic study programs. 
            You analyze a data structure describing the study program, according to the detailed instructions provided in the first message, in order to answer student questions about this program.

            The JSON file with the study program data is provided in the code interpreter tool.
            You may also align the study program with student grade lists, and suggest courses the student should take next. 
            The language of most of the content is Hebrew. Hebrew names for elements such as course names, faculty names, study program names, etc. must be provided verbatim, exactly as given in the provided content (returned by the provided tools).
            Your answer should be in the same language as the query - if the user's question is in Hebrew, your answer should be in Hebrew.
            Study programs tend to change over time, and footnotes are used to address students who took courses from previous versions of the same study program.
            You can perform the following:
            1. Parse and understand study program details from the JSON file, according to the detailed instructions provided.
            2. Answer questions about the study program.
            3. If the student provides a course grade report, check the student's completed courses against the program requirements, and suggest next courses to fulfill mandatory and elective requirements.
            4. Always consider course prerequisites and constraints.

            Respond in a clear, structured format with detailed reasoning.
        """

        self.program_data = self._load_json_file("cs_study_programs.json")
        for program in self.program_data:
            self.study_programs[program['code']] = program['name']

        # Read text file containing instructions
        self.dir_path = self.config["DB_Path"]
        text_path = os.path.join(self.dir_path, "Study_Program_json_guide.txt")
        with open(text_path, "r", encoding='utf-8') as text_file:
            self.program_instructions = text_file.read()

    ##############################################################################
    def get_system_instructions(self):
        return [self.system_instructions, self.program_instructions]

    ##############################################################################
    def _init_tools(self):

       ##############################################################################
        class ProgramCourse(BaseModel):
            type: str

        class SimpleCourse(ProgramCourse):
            id: str
            name: str
            url: str
            level: str
            points: int
            footnotes: List[str]

        class CompoundCourse(ProgramCourse):
            operator: str
            children: List[ProgramCourse]

        class SubSection(BaseModel):
            title: str
            footnotes: List[str]
            text: List[str]
            courses: List[ProgramCourse]

        class Section(BaseModel):
            title: str
            footnotes: List[str]
            text: List[str]
            sub_sections: List[SubSection]
            courses: List[ProgramCourse]

        class StudyProgramDetails(BaseModel):
            code: str
            name: str
            text: List[str]
            footnotes: List[str]
            sections: List[Section]

        ##############################################################################
        @tool("GetStudyProgramDetailsFromCode")
        def _get_study_program_details_from_code(study_program_code: str) -> StudyProgramDetails:
            """Get the study program details from the study program code.
            
            Args:
                study_program_code: The code of the study program to look up
            """

            if study_program_code not in self.study_programs:
                print(f"In Tool: Study program {study_program_code} not found")
                return None
            
            result = self.study_programs[study_program_code]
            print(f"In Tool: Getting study program details for {study_program_code}\n")
            return result
        
        ##############################################################################
        self.tools = [
                    _get_study_program_details_from_code,
                ]

    ##############################################################################
    def get_prompt(self, study_program_code, query_text):
        prompt = """
            Using study program code {study_program_code}, analyze the following user query:
            {query_text}
            """

        filled_prompt = prompt.format(
            study_program_code=study_program_code,
            query_text=query_text
        )
        
        return filled_prompt

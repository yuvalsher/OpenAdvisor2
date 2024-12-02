import openai
from openai import OpenAI
import dotenv
import os
import json
from typing_extensions import override
from openai import AssistantEventHandler

from AbstractAgent import AbstractAgent
from config import all_config

# Define system instructions
system_instructions = """
You are an academic advisor specializing in study programs. You analyze JSON files describing study programs, align them with student grade lists, and suggest courses the student should take next. 
You can perform the following:
1. Parse and understand study program details from the JSON file.
2. Answer questions about the study program.
3. Check the student's completed courses against the program requirements.
4. Suggest next courses to fulfill mandatory and elective requirements.
5. Always consider course prerequisites and constraints.

Respond in a clear, structured format with detailed reasoning.
"""

class OpenAIAssistant(AbstractAgent):

    ##############################################################################
    def __init__(self, config):
        super().__init__(config)
        dotenv.load_dotenv()
        self.openai = OpenAI()
        self.openai.api_key = os.getenv("OPENAI_API_KEY")
        self.assistants = {}

    ##############################################################################
    def _init_tools(self):
        pass

    ##############################################################################
    def _init_data(self):

        dir_path = self.config["DB_Path"]

        # Load JSON data
        json_path = os.path.join(dir_path, "cs_study_programs.json")
        with open(json_path, "r", encoding='utf-8') as json_file:
            data = json.load(json_file)
        self.study_programs_data = {}
        for faculty in data:
            code = faculty['code']
            self.study_programs_data[code] = faculty
        # Read text file containing instructions
        text_path = os.path.join(dir_path, "Study_Program_json_guide.txt")
        with open(text_path, "r", encoding='utf-8') as text_file:
            self.program_instructions = text_file.read()

    ##############################################################################
    def create_assistant(self, faculty_code, instructions, model):

        if faculty_code not in self.study_programs_data:
            print(f"Faculty code {faculty_code} not found in study programs data")
            return None

        # Format JSON data for inclusion in the system message
        formatted_study_programs_data = json.dumps(self.study_programs_data[faculty_code], indent=2)

        # Combine instructions and data
        system_message_content = f"{system_instructions}\n\n{self.program_instructions}\n\nData on Academic Study Programs:\n{formatted_study_programs_data}"

        name = f"Study Advisor for {faculty_code}"
        tools=[{"type": "code_interpreter"}]
        assistant = self.openai.beta.assistants.create(
            name=name,
            instructions=system_message_content,
            tools=tools,
            model=model,
        )
        return assistant
    
    ##############################################################################
    def get_assistant(self, faculty_code):
        if faculty_code not in self.assistants:
            self.assistants[faculty_code] = self.create_assistant(faculty_code, system_instructions, "gpt-4o-mini")
        return self.assistants[faculty_code]
    
    ##############################################################################
    def create_thread(self, query):
        thread = self.openai.beta.threads.create()
    
        # Add the query to the Thread
        message = self.openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=query
        )
        
        return thread
    
    ##############################################################################
    def create_run_and_wait(self, thread_id, assistant_id, input):
        run = self.openai.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            input=input,
        )
        if run.status == 'completed': 
            messages = self.openai.beta.threads.messages.list(thread_id=thread_id)
            return messages.data[0].content[0].text.value
        else:
            return None

    ##############################################################################
    def create_run_stream(self, thread_id, assistant_id, instructions):
        answer = ""
        class EventHandler(AssistantEventHandler):    
            @override
            def on_text_created(self, text) -> None:
                global answer
                answer = str(text)
                print(f"\nassistant > ", end="", flush=True)
            
            @override
            def on_text_delta(self, delta, snapshot):
                global answer
                answer += delta.value
                print(delta.value, end="", flush=True)
            
            def on_tool_call_created(self, tool_call):
                print(f"\nassistant > {tool_call.type}\n", flush=True)
            
            def on_tool_call_delta(self, delta, snapshot):
                if delta.type == 'code_interpreter':
                    if delta.code_interpreter.input:
                        print(delta.code_interpreter.input, end="", flush=True)
                    if delta.code_interpreter.outputs:
                        print(f"\n\noutput >", flush=True)
                        for output in delta.code_interpreter.outputs:
                            if output.type == "logs":
                                print(f"\n{output.logs}", flush=True)

        with self.openai.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
            instructions=instructions,
            event_handler=EventHandler(),
        ) as stream:
            stream.until_done()
            responses = stream.get_final_messages()
            return responses[0].content[0].text.value

##############################################################################
def main(fac_code):
    query = "כמה נקודות זכות צריך להשלים עבור קורסי בחירה?"
    ass_ids = {
        "AF": "asst_FhECPescLvwXFIynJfrOgpi9",
    }
    
    ass_agent = OpenAIAssistant(all_config["General"])
    ass_agent.init()

    if fac_code in ass_ids:
        ass_id = ass_ids[fac_code]
        print(f"Assistant ID for {fac_code}: {ass_id}")
    else:
        ass = ass_agent.get_assistant(fac_code)
        ass_id = ass.id
        print(f"Assistant ID for {fac_code}: {ass_id}")

    ### Create a thread
    thread = ass_agent.create_thread(query)


    answer = ass_agent.create_run_stream(thread.id, ass_id, "")
    print(f"Answer: {answer[::-1]}")

    # response = ass_agent.create_run_and_wait(thread.id, ass_id, query)

##############################################################################
if __name__ == "__main__":
    main("AF")

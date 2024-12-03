import openai
from openai import OpenAI
import dotenv
import os
import json
from typing_extensions import override
from openai import AssistantEventHandler

from AbstractAgent import AbstractAgent
from config import all_config

class OpenAIAssistant(AbstractAgent):

    # Define system instructions
    system_instructions = """
        You are an academic advisor for the Open University of Israel (האוניברסיטה הפתוחה), specializing in academic study programs. 
        You analyze a JSON file describing the study program, according to the detailed instructions provided in the first message, in order to answer student questions about this program.
        The JSON file with the study program data is provided in the first message, use the code interpreter tool to read it.
        You may also align the study program with student grade lists, and suggest courses the student should take next. 
        The language of most of the content is Hebrew. Hebrew names for elements such as course names, faculty names, study program names, etc. must be provided verbatim, exactly as given in the provided content (returned by the provided tools).
        You can perform the following:
        1. Parse and understand study program details from the JSON file, according to the detailed instructions provided.
        2. Answer questions about the study program.
        3. If the student provides a course grade report, check the student's completed courses against the program requirements, and suggest next courses to fulfill mandatory and elective requirements.
        4. Always consider course prerequisites and constraints.

        Respond in a clear, structured format with detailed reasoning.
    """

    ##############################################################################
    def __init__(self, config):
        super().__init__(config)
        dotenv.load_dotenv()
        self.openai = OpenAI()
        self.openai.api_key = os.getenv("OPENAI_API_KEY")
        self.assistant = None  # We'll store just one assistant

    ##############################################################################
    def _init_tools(self):
        pass

    ##############################################################################
    def _init_data(self):
        dir_path = self.config["DB_Path"]
        # Read text file containing instructions
        text_path = os.path.join(dir_path, "Study_Program_json_guide.txt")
        with open(text_path, "r", encoding='utf-8') as text_file:
            self.program_instructions = text_file.read()

    ##############################################################################
    def get_assistant(self):
        if self.assistant is None:
            # Try to find existing assistant
            name = "Study Program Advisor"
            assistants_list = self.openai.beta.assistants.list(
                order="desc",
                limit=100
            )
            
            # Look for an existing assistant with this name
            for assistant in assistants_list.data:
                if assistant.name == name:
                    print(f"Found existing assistant: {assistant.id}")
                    self.assistant = assistant
                    return self.assistant

            # Create new assistant if none exists
            self.assistant = self.openai.beta.assistants.create(
                name=name,
                instructions=self.system_instructions,
                tools=[{"type": "code_interpreter"}],
                model="gpt-4-turbo-preview",
            )
            print(f"Created new assistant: {self.assistant.id}")
        
        return self.assistant

    ##############################################################################
    def create_thread(self, faculty_code, query):
        # Read the JSON file content
        dir_path = self.config["DB_Path"]
        json_path = os.path.join(dir_path, "study_programs", faculty_code+".json")
        with open(json_path, 'r', encoding='utf-8') as f:
            json_content = f.read()

        # Create initial messages with both the program instructions and JSON content
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Here are the instructions for understanding the study program data:
                        {self.program_instructions}
                        
                        The study program data in JSON format is:
                        {json_content}
                        
                        Please confirm you can read and parse this JSON data before proceeding."""
                    },
                ],
            },
        ]

        thread = self.openai.beta.threads.create(messages=messages)
        
        # Add the user's query to the Thread
        message = self.openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=query
        )
        
        return thread
    
    ##############################################################################
    def create_run_and_wait(self, thread_id, assistant_id, instructions):
        run = self.openai.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=assistant_id,
            instructions=instructions,
            poll_interval_ms=500,
        )
        if run.status == 'completed': 
            messages = self.openai.beta.threads.messages.list(thread_id=thread_id)
            return messages.data[0].content[0].text.value
        else:
            return None

    ##############################################################################
    def create_run_stream(self, thread_id, assistant_id, instructions):
        class EventHandler(AssistantEventHandler):    
            @override
            def on_text_created(self, text) -> None:
                print(f"\nassistant > ", end="", flush=True)
            
            @override
            def on_text_delta(self, delta, snapshot):
                global answer
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
        "AF": "asst_OLknkJI8eLDH7izjmHFtVXw9",
    }
    
    ass_agent = OpenAIAssistant(all_config["General"])
    ass_agent.init()

    if fac_code in ass_ids:
        ass_id = ass_ids[fac_code]
    else:
        ass = ass_agent.get_assistant()
        ass_id = ass.id
        print(f"Assistant ID for {fac_code}: {ass_id}")

    ### Create a thread
    thread = ass_agent.create_thread(fac_code, query)


    #answer = ass_agent.create_run_stream(thread.id, ass_id, "")
    answer = ass_agent.create_run_and_wait(thread.id, ass_id, "")
    print(f"Answer: {answer[::-1]}")

    # response = ass_agent.create_run_and_wait(thread.id, ass_id, query)

##############################################################################
if __name__ == "__main__":
    main("AF")

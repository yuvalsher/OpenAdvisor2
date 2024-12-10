from typing import Dict
import re
import openai
from openai import OpenAI
import dotenv
import os
import json
from typing_extensions import override
from openai import AssistantEventHandler
from langchain.memory import ConversationBufferMemory

from config import all_config


USER_MESSAGE      = "user"
ASSISTANT_MESSAGE = "assistant"

class OpenAIAssistant():

    # Define system instructions
    system_instructions = """
        You are an academic advisor for the Open University of Israel (האוניברסיטה הפתוחה), specializing in academic study programs. 
        You analyze a JSON file describing the study program, according to the detailed instructions provided in the first message, in order to answer student questions about this program.
        The JSON data of the study program is provided in one of the first messages, and should be parsed using the code interpreter tool.
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

    ##############################################################################
    def __init__(self, config):
        self.config = config
        dotenv.load_dotenv()
        self.openai = OpenAI()
        self.openai.api_key = os.getenv("OPENAI_API_KEY")
        self.threads: Dict[str, "Thread"] = {}

    ##############################################################################
    def init(self):
        self._init_tools()
        self._init_data()

    ##############################################################################
    def _init_tools(self):
        pass

    ##############################################################################
    def _init_data(self):
        self.dir_path = self.config["DB_Path"]

        # Read text file containing instructions
        text_path = os.path.join(self.dir_path, "Study_Program_json_guide.txt")
        with open(text_path, "r", encoding='utf-8') as text_file:
            self.program_instructions = text_file.read()

        # Load JSON data
        json_path = os.path.join(self.dir_path, "cs_study_programs.json")
        with open(json_path, "r", encoding='utf-8') as json_file:
            data = json.load(json_file)
        self.study_programs_data = {}
        for faculty in data:
            code = faculty['code']
            self.study_programs_data[code] = faculty

    ##############################################################################
    def get_assistant(self):
        # Try to find existing assistant
        name = "OUI Study Program Advisor"
        assistants_list = self.openai.beta.assistants.list(
            order="desc",
            limit=100
        )
        
        # Look for an existing assistant with this name
        for assistant in assistants_list.data:
            if assistant.name == name:
                print(f"Found existing assistant: {assistant.id}")
                return assistant

        assistant = self.openai.beta.assistants.create(
            name=name,
            instructions=self.system_instructions,
            tools=[{"type": "code_interpreter"}],
            model="gpt-4o-mini",
            temperature=0.3
        )
        print(f"Created new assistant: {assistant.id}")
        
        return assistant

    ##############################################################################
    def add_message(self, thread_id, role, content):
        self.openai.beta.threads.messages.create(
            thread_id=thread_id,
            role=role,
            content=content
        )

    ##############################################################################
    def create_thread(self, faculty_code):
        thread = self.openai.beta.threads.create()

        # Load the instructions to understand the JSON file
        self.add_message(thread.id, USER_MESSAGE, f"Here are the instructions for understanding the study program data: \n{self.program_instructions}")
        json_data = json.dumps(self.study_programs_data[faculty_code], indent=2)
        self.add_message(thread.id, USER_MESSAGE, f"This is the definition of the study program in JSON format:\n{json_data}")

        return thread

    ##############################################################################
    def create_run_and_wait(self, thread_id, assistant_id):
        run = self.openai.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )
        print ("Sending request...")
        if run.status == 'completed': 
            messages = self.openai.beta.threads.messages.list(thread_id=thread_id)
            print ("Request completed")
            return messages.data[0].content[0].text.value
        else:
            print ("Request not completed")
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
    def _get_or_create_thread(self, client_id: str, chat_history: ConversationBufferMemory) -> "Thread":
        """Get existing thread for client_id or create new if doesn't exist."""
        if client_id in self.threads:
            return self.threads[client_id]
        else:
            thread = self.create_thread(chat_history)
            self.threads[client_id] = thread
            return thread

    ##############################################################################
    def do_query(self, user_input: str, faculty_code: str) -> tuple[str, str]:
        """
        Process a query using multiple agents.
        
        Args:
            user_input: The user's query
            faculty_code: The faculty code of the study program
            
        Returns:
            str: response_text
        """

        print(f"Entering OpenAI Assistant2 for {faculty_code}: user_input: {user_input[::-1]}")
        if faculty_code not in self.study_programs_data:
            msg =f"Error: Faculty code {faculty_code} not found in study programs data"
            print(msg)
            return msg

        assistant = self.get_assistant()
        thread = self.create_thread(faculty_code)
        self.add_message(thread.id, USER_MESSAGE, user_input)
        answer = self.create_run_and_wait(thread.id, assistant.id)
        print_answer(answer)

        return answer

    ##############################################################################
    def reset_chat_history(self, client_id: str):
        """
        Reset chat history for a specific client.
        
        Args:
            client_id: The client's unique identifier
        """
        if client_id in self.threads:
            new_thread = self.create_thread(None)
            self.threads[client_id] = new_thread

##############################################################################
def print_answer(answer):
        # Regular expression pattern to match Hebrew characters
    hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
    # Count total alphabetic characters
    total_alpha_chars = sum(1 for char in answer if char.isalpha())
    # Count Hebrew characters
    hebrew_chars = len(hebrew_pattern.findall(answer))
    # Determine if the majority are Hebrew
    if (hebrew_chars > total_alpha_chars / 2):
        # Mostly Hebrew characters, print reversed
        print(f"Answer: {answer[::-1]}")
    else:
        # No Hebrew, print normally
        print(f"Answer: {answer}")


##############################################################################
def main(fac_code):
    query1 = "איזה קורסי מתמטיקה הם חלק מהתוכנית?"
    query2 = "כמה נקודות זכות נותן הקורס השני?"  
    ass_agent = OpenAIAssistant(all_config["General"])
    ass_agent.init(fac_code)

    ass = ass_agent.get_assistant(fac_code)

    ### Create a thread
    thread = ass_agent.create_thread()

    #answer = ass_agent.create_run_stream(thread.id, ass_id, "")
    ass_agent.add_user_message(thread.id, query1)
    answer = ass_agent.create_run_and_wait(thread.id, ass.id)
    print_answer(answer)

    ass_agent.add_user_message(thread.id, query2)
    answer = ass_agent.create_run_and_wait(thread.id, ass.id)
    print_answer(answer)

##############################################################################
if __name__ == "__main__":
    main("AF")

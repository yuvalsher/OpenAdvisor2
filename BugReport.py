import openai
from openai import OpenAI
import dotenv
import os
import json
from typing_extensions import override
from openai import AssistantEventHandler

class OpenAIAssistant():

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
    def init(self):
        dotenv.load_dotenv()
        self.openai = OpenAI()
        self.openai.api_key = os.getenv("OPENAI_API_KEY")
        self.assistant = None

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.dir_path = os.path.join(script_dir, "kb", "json_source")
        # Read text file containing instructions
        text_path = os.path.join(self.dir_path, "Study_Program_json_guide.txt")
        with open(text_path, "r", encoding='utf-8') as text_file:
            self.program_instructions = text_file.read()

    ##############################################################################
    def get_assistant(self):
        if self.assistant is None:
            # Try to find existing assistant
            name = "BugReportExample"
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

            file = self.openai.files.create(
                file=open(os.path.join(self.dir_path, "study_programs", "AF.json"), "rb"),
                purpose='assistants'
            )

            # Create new assistant if none exists
            self.assistant = self.openai.beta.assistants.create(
                name=name,
                instructions=self.system_instructions,
                tools=[{"type": "code_interpreter"}],
                tool_resources={
                    "code_interpreter": {
                        "file_ids": [file.id]
                    }
                },
                model="gpt-4-turbo-preview",
            )
            print(f"Created new assistant: {self.assistant.id}")
        
        return self.assistant

    ##############################################################################
    def create_thread(self, query):
        # Create initial messages with both the program instructions and JSON content
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Here are the instructions for understanding the study program data:
                        {self.program_instructions}
                        """
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
def main():
    #query = "how many credit points do I need to complete elective courses?"
    query = "how many math courses are needed to complete the program?"
    
    ass_agent = OpenAIAssistant()
    ass_agent.init()

    ass = ass_agent.get_assistant()

    ### Create a thread
    thread = ass_agent.create_thread(query)


    answer = ass_agent.create_run_and_wait(thread.id, ass.id, "")
    print(f"Answer: {answer}")


##############################################################################
if __name__ == "__main__":
    main()

import openai
from openai import OpenAI
import dotenv
import os
import json

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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

def create_assistant(name, instructions, tools, model):
    client = OpenAI()
    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Write and run code to answer math questions.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4o",
    )
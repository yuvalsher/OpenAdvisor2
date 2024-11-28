import logging
import sys
import os
from dash_chat import DashChat
from langtrace_python_sdk import langtrace


from rag import Rag
from MultiAgent import MultiAgent
from CoursesWithTools import CoursesWithTools
from config import all_config


##############################################################################
# Redirect print statements to logging
class PrintToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

##############################################################################
def main(type, faculty_code):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Change this to logging.INFO to reduce debug messages
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log', encoding='utf-8')
        ]
    )

    # Disable debug logging for specific modules
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)

    langtrace.init(api_key=os.getenv("LANGTRACE_API_KEY"))
    
    # Fetch the port from the environment, default to 10000
    port = int(os.getenv('PORT', 10000))
    general_config = all_config["General"]

    llm_obj = None
    if type == "MultiAgent":
        llm_obj = MultiAgent(general_config)
    elif type == "Tools":
        llm_obj = CoursesWithTools(general_config)
    else:
        llm_obj = Rag(general_config)

    llm_obj.init(faculty_code)

    dash_chat = DashChat(llm_obj)
    title = general_config["title"]
    subtitle = general_config["description"]
    dash_chat.init(title, subtitle, general_config)

    # Change this line to bind to all network interfaces
    dash_chat.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    main("MultiAgent", "")
    #main("Tools", "CS")
    #main("RAG", "Courses")

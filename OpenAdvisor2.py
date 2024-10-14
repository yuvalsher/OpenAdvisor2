import logging
import sys
import os
from dash_chat import DashChat
from rag import Rag
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
def main(faculty_code):
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log', encoding='utf-8')
        ]
    )

    # Create a logger that can handle Unicode
    logger = logging.getLogger()
    sys.stdout = PrintToLogger(logger)
    sys.stderr = PrintToLogger(logger, log_level=logging.ERROR)
    
    # Fetch the port from the environment, default to 10000
    port = int(os.getenv('PORT', 10000))
    text_config = all_config[faculty_code]
    general_config = all_config["General"]

    rag = Rag()
    rag.init(faculty_code, general_config)

    dash_chat = DashChat(rag)
    title = text_config["title"]
    subtitle = text_config["description"]
    dash_chat.init(title, subtitle, general_config)

    # Change this line to bind to all network interfaces
    dash_chat.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    main("CS")

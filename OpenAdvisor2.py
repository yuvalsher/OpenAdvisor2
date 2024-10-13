import logging
import sys
import os
from dash_chat import DashChat
from rag import Rag


all_config = {
    "OUI": {
        "title": "האוניברסיטה הפתוחה - ייעוץ כללי",
        "description": "אני בוט הייעוץ הכללי של האוניברסיטה הפתוחה. תוכלו לשאול אותי שאלות על הלימודים באוניברסיטה הפתוחה שאינן ספציפיות לתוכנית לימודים כזו או אחרת."
    },
    "CS": {
        "title": "האוניברסיטה הפתוחה - ייעוץ למדעי המחשב",
        "description": "אני בוט הייעוץ של הפקולטה למדעי המחשב של האוניברסיטה הפתוחה. תוכלו לשאול אותי שאלות על הלימודים בפקולטה."
    }
}

##############################################################################
# Redirect print statements to logging
class PrintToLogger:
    def write(self, message):
        if message.strip():  # Avoid logging empty messages
            logging.info(message.strip())

    def flush(self):
        pass

##############################################################################
def main(faculty_code):

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Log to standard output
            logging.FileHandler('app.log')      # Log to a file
        ]
    )

    sys.stdout = PrintToLogger()
    
    # Fetch the port from the environment, default to 10000
    port = int(os.getenv('PORT', 10000))
    config = all_config[faculty_code]

    rag = Rag()
    rag.init(faculty_code)

    dash_chat = DashChat(rag)
    title = config["title"]
    subtitle = config["description"]
    dash_chat.init(title, subtitle)

    # Change this line to bind to all network interfaces
    dash_chat.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    main("CS")

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
def main(faculty_code):
    # Fetch the port from the environment
    port = int(os.getenv('PORT', 10000))
    config = all_config[faculty_code]

    rag = Rag()
    rag.init(faculty_code)

    dash_chat = DashChat(rag)
    title = config["title"]
    subtitle = config["description"]
    dash_chat.init(title, subtitle)

    dash_chat.run(port)

if __name__ == "__main__":
    main("CS")
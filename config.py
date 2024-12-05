import os
from dotenv import load_dotenv, dotenv_values

# Function to load environment variables and return the configuration
def get_all_config():
    # Load environment variables from .env file
    load_dotenv()
    env_vars = dotenv_values()

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    all_config = {
        "General": {
            "title": "האוניברסיטה הפתוחה - ייעוץ אקדמי",
            "description": "אני בוט הייעוץ האקדמי של האוניברסיטה הפתוחה. תוכלו לשאול אותי שאלות על הלימודים באוניברסיטה.\nהקש 'נקה' לאיפוס היסטוריית השיחה.",
            "embeddings": "text-embedding-3-small",
            "llm_model": "gpt-4o-mini",
            "router_llm_model": "gpt-4o-mini",
            "Chroma_Path": os.path.join(script_dir, "kb", "chroma"),
            "DB_Path": os.path.join(script_dir, "kb", "json_source"),
            "Rag_System_message": "You are an AI model serving as an academic advisor for the Open University of Israel (OUI). The name of the OUI in Hebrew is האוניברסיטה הפתוחה. ",
            "Chat_Welcome_Message": "שלום, אני עוזר וירטואלי של האוניברסיטה הפתוחה. כיצד אוכל לעזור לך?",
            "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY'),
            "ANTHROPIC_API_KEY": os.getenv('ANTHROPIC_API_KEY'),
            "GOOGLE_API_KEY": os.getenv('GOOGLE_API_KEY'),
            "GOOGLE_CSE_ID": os.getenv('GOOGLE_CSE_ID'),
            "LANGCHAIN_API_KEY": os.getenv('LANGCHAIN_API_KEY'),
            "AI21_API_KEY": os.getenv('AI21_API_KEY'),
            "msg_sender_field": "role",
            "msg_text_field": "content",
            "user_name": "user",
            "bot_name": "assistant"
        },
        "OUI": {
            "title": "האוניברסיטה הפתוחה - ייעוץ כללי",
            "description": "אני בוט הייעוץ הכללי של האוניברסיטה הפתוחה. תוכלו לשאול אותי שאלות על הלימודים באוניברסיטה הפתוחה שאינן ספציפיות לתוכנית לימודים כזו או אחרת."
        },
        "CS": {
            "title": "האוניברסיטה הפתוחה - ייעוץ למדעי המחשב",
            "description": "אני בוט הייעוץ של הפקולטה למדעי המחשב של האוניברסיטה הפתוחה. תוכלו לשאול אותי שאלות על הלימודים בפקולטה."
        },
        "Courses": {
            "title": "האוניברסיטה הפתוחה - מידע על קורסים",
            "description": "אני בוט הייעוץ של האוניברסיטה הפתוחה. תוכלו לשאול אותי שאלות על קורסים באוניברסיטה הפתוחה."
        }
    }

    return all_config

# Initialize the configuration
all_config = get_all_config()

all_crawl_config = {
    "All": {
        "start_urls": [
            'https://www.openu.ac.il/counseling/Pages/Preparation_for_counselin.aspx',
            'https://www.openu.ac.il/registration/pages/default.aspx',
            'https://www.openu.ac.il/deanacademicstudies/teachandlearn/learning_skiils/pages/default.aspx',
            'https://www.openu.ac.il/students/pages/default.aspx',
            'https://www.openu.ac.il/library/pages/default.aspx'
        ],
        "allowed_domains": [
            'academic.openu.ac.il',
            'www.openu.ac.il', 
        ],
        "disallowed_domains": [
            'academic.openu.ac.il/yedion',
            'academic.openu.ac.il/degrees/Pages',
            'www.openu.ac.il/dean-students/opjob/events',
            'www.openu.ac.il/publications',
            'www.openu.ac.il/oui-press',
            'www.openu.ac.il/events',
            'www.openu.ac.il/allnews',
            'www.openu.ac.il/personal_sites',
            'www.openu.ac.il/jcmcenter',
            'www.openu.ac.il/library/New_in_the_library',
            'www.openu.ac.il/library/new_in_the_library',
            'www.openu.ac.il/staff/pages/results.aspx',
            'www.openu.ac.il/about/Pages/procedures.aspx'
        ],
        "disallowed_pages": [
        ]
    },
    "OUI": {
        # Define the initial URLs to start crawling from
        "start_urls": [
            'https://www.openu.ac.il/counseling/Pages/Preparation_for_counselin.aspx',
            'https://www.openu.ac.il/registration/pages/default.aspx'
            'https://www.openu.ac.il/deanacademicstudies/teachandlearn/learning_skiils/pages/default.aspx',
            'https://www.openu.ac.il/students/pages/default.aspx',
            'https://www.openu.ac.il/library/pages/default.aspx'
        ],
        # Define the allowed domains to limit the crawler
        "allowed_domains": [
            'www.openu.ac.il/counseling', 
            'www.youtube.com/watch',
            'academic.openu.ac.il/more-than-degree/ba',
            'academic.openu.ac.il/degrees',
            'academic.openu.ac.il/english',
            'academic.openu.ac.il/yedion',
            'www.openu.ac.il/young',
            'www.openu.ac.il/dean-students',
            'www.openu.ac.il/ar',
            'www.openu.ac.il/registration'
            'www.openu.ac.il/deanacademicstudies',
            'www.openu.ac.il/students',
            'www.openu.ac.il/library'
        ],
        "disallowed_domains": [
            'academic.openu.ac.il/yedion',
            'academic.openu.ac.il/degrees/Pages',
            'www.openu.ac.il/dean-students/opjob/events'
        ],
        "disallowed_pages": [
            'https://academic.openu.ac.il/degrees/Pages/archive.aspx',
            'https://www.openu.ac.il/dean-students/involvement_and_sport/Pages/association_volunteer.aspx',
            'https://www.openu.ac.il/dean-students/involvement_and_sport/pages/overseas.aspx',
            'https://www.openu.ac.il/dean-students/involvement_and_sport/pages/Dean_Award_2019.aspx',
            'https://www.openu.ac.il/dean-students/involvement_and_sport/pages/Dean_Award_2018.aspx',
            'https://www.openu.ac.il/dean-students/accessibility/Pages/negishop.aspx',
            'https://www.openu.ac.il/dean-students/accessibility/Pages/negishop_lobby.aspx',
            'https://www.openu.ac.il/dean-students/Scholarships/Pages/1000Scholarships_old.aspx'
        ]
    },
    "CS": {
        "start_urls": [
            'https://academic.openu.ac.il/cs/computer/pages/default.aspx'
        ],
        "allowed_domains": [
            'academic.openu.ac.il/cs/computer'
        ],
        "disallowed_domains": [
        ],
        "disallowed_pages": [
        ]
    },
    "Math": {
        "start_urls": [
            'https://academic.openu.ac.il/cs/mathematics/pages/default.aspx'
        ],
        "allowed_domains": [
            'academic.openu.ac.il/cs/mathematics'
        ],
        "disallowed_domains": [
        ],
        "disallowed_pages": [
        ]
    },
    "Industrial": {
        "start_urls": [
            'https://academic.openu.ac.il/cs/industrial_engineering/pages/default.aspx'
        ],
        "allowed_domains": [
            'academic.openu.ac.il/cs/industrial_engineering'
        ],
        "disallowed_domains": [
        ],
        "disallowed_pages": [
        ]
    },
    
}

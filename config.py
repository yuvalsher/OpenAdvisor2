import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the OpenAI API key from environment variables or use the default
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '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')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyAkvi4mxoMZgTb8yDM3bWc8rfjtLrcCwwo')

all_config = {
    "General": {
        "embeddings": "text-embedding-3-small",
        "llm_model": "gpt-4o-mini",
        "Chroma_Path": os.path.join(script_dir, "kb", "chroma"),
        "DB_Path": os.path.join(script_dir, "kb", "json_source"),
        "Rag_System_message": "You are an AI model serving as an academic advisor for the Open University of Israel (OUI). The name of the OUI in Hebrew is האוניברסיטה הפתוחה. ",
        "Chat_Welcome_Message": "שלום, איך אוכל לעזור לך היום?",
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "GOOGLE_API_KEY": GOOGLE_API_KEY
    },
    "OUI": {
        "title": "האוניברסיטה הפתוחה - ייעוץ כללי",
        "description": "אני בוט הייעוץ הכללי של האוניברסיטה הפתוחה. תוכלו לשאול אותי שאלות על הלימודים באוניברסיטה הפתוחה שאינן ספציפיות לתוכנית לימודים כזו או אחרת."
    },
    "CS": {
        "title": "האוניברסיטה הפתוחה - ייעוץ למדעי המחשב",
        "description": "אני בוט הייעוץ של הפקולטה למדעי המחשב של האוניברסיטה הפתוחה. תוכלו לשאול אותי שאלות על הלימודים בפקולטה."
    }
}

all_crawl_config = {
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
    
}

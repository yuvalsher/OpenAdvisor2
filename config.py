import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the OpenAI API key from environment variables or use the default
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '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')

all_config = {
    "General": {
        "embeddings": "text-embedding-3-small",
        "llm_model": "gpt-4o-mini",
        "Chroma_Path": os.path.join(script_dir, "kb", "chroma"),
        "DB_Path": os.path.join(script_dir, "kb", "json_source"),
        "Rag_System_message": "You are an AI model serving as an academic advisor for the Open University of Israel (OUI). The name of the OUI in Hebrew is האוניברסיטה הפתוחה. ",
        "Chat_Welcome_Message": "שלום, איך אוכל לעזור לך היום?",
        "OPENAI_API_KEY": OPENAI_API_KEY
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

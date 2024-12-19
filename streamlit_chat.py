import os
import sys
import logging
from uuid import uuid4
from langtrace_python_sdk import langtrace
import streamlit as st
from openai import OpenAI

from config import all_config
from MultiAgent2 import MultiAgent2

##############################################################################
def init_session_state(config, llm_obj):
    # Initialize all session state variables if they don't exist
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "processing" not in st.session_state:
        st.session_state["processing"] = False
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""
    if "current_input" not in st.session_state:
        st.session_state["current_input"] = ""
    if "title" not in st.session_state:
        st.session_state["title"] = config["title"]
    if "subtitle" not in st.session_state:
        st.session_state["subtitle"] = config["description"]
    if "llm_obj" not in st.session_state:
        st.session_state["llm_obj"] = llm_obj
    if "client_id" not in st.session_state:
        st.session_state["client_id"] = str(uuid4())

##############################################################################
def clear_input():
    st.session_state.input_text = ""

##############################################################################
@st.cache_resource
def init():
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
#    port = int(os.getenv('PORT', 10000))
    general_config = all_config["General"]

    llm_obj = MultiAgent2(general_config)
    llm_obj.init("All")

    return general_config, llm_obj

##############################################################################
def init_css():
    # Custom CSS for styling
    st.markdown("""
        <style>
        body, html {
            direction: RTL;
            unicode-bidi: bidi-override;
            text-align: right;
        }
        p, div, input, label, h1, h2, h3, h4, h5, h6 {
            direction: RTL;
            unicode-bidi: bidi-override;
            text-align: right;
        }
        /* Main container styles */
        .main {
            max-width: 100%;
            padding: 0;
        }
        
        /* Chat container styles */
        .chat-container {
            height: 70vh;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
            background-color: #f9f9f9;
            direction: rtl;
            text-align: right;
        }
        
        /* Message styles */
        .chat-message {
            padding: 0.5rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            max-width: 85%;
            direction: rtl;
            text-align: right;
        }
        .chat-message.user {
            background-color: #e6f3ff;
            margin-left: auto;
        }
        .chat-message.bot {
            background-color: #f0f0f0;
            margin-right: auto;
        }
        
        /* Input area styles */
        .input-area {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 1rem;
            background-color: white;
            border-top: 1px solid #ddd;
            direction: rtl;
            text-align: right;
        }
        
        /* Button styles */
        .stButton button {
            width: 100%;
            border-radius: 5px;
            height: 2.5em;
        }
        
        /* Adjust main content padding to prevent overlap with fixed input area */
        .main .block-container {
            padding-bottom: 180px;
        }
        
        /* Disabled button styles */
        .stButton button:disabled {
            background-color: #e0e0e0;
            cursor: wait;
            opacity: 0.7;
        }

        /* Loading cursor for the whole page when processing */
        .processing {
            cursor: wait !important;
        }
                
        /* File uploader styles */
        [data-testid='stFileUploader'] {
            width: max-content;
        }
        [data-testid='stFileUploader'] section {
            padding: 0;
            float: left;
        }
        [data-testid='stFileUploader'] section > input + div {
            display: none;
        }
        [data-testid='stFileUploader'] section + div {
            float: right;
            padding-top: 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Add processing class to body if processing
    if st.session_state["processing"]:
        st.markdown("""
            <script>
                document.body.classList.add('processing');
            </script>
        """, unsafe_allow_html=True)

##############################################################################
def main():
    # Set page config
    st.set_page_config(
        page_title="האוניברסיטה הפתוחה - ייעוץ אקדמי",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    config, llm_obj = init()
    init_session_state(config, llm_obj)
    init_css()

    # Header
    st.title(config["title"])
    st.caption(config["description"])

    # Create a container for chat history
    chat_container = st.container()
    # Create a container for input elements at the bottom
    input_container = st.container()

    # Display chat history in the chat container
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Use the input container for input elements
    with input_container:
        # Create columns for file upload and chat input (reversed order for RTL)
        input_col, upload_col = st.columns([6, 1])  # Reversed order
        
        with input_col:
            if prompt := st.chat_input("What is up?"):
                # Add user message to session state
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state["current_input"] = prompt  # Store the input for processing
                st.session_state["processing"] = True  # Set processing flag
                st.rerun()  # Rerun to display user message immediately

            # Check if we have a pending message to process
            if "current_input" in st.session_state and st.session_state["processing"]:
                # Get bot response
                bot_response, client_id = llm_obj.do_query(st.session_state["current_input"], st.session_state["messages"], st.session_state["client_id"])
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                del st.session_state["current_input"]  # Clear the pending input
                st.session_state["processing"] = False  # Reset processing flag
                st.rerun()  # Rerun to update with the bot response
        
        with upload_col:
            uploaded_file = st.file_uploader("Upload", 
                                         key="upload_button",
                                         type=["pdf"],
                                         label_visibility="collapsed",
                                         disabled=st.session_state["processing"])

    # Handle file upload
    if uploaded_file:
        st.success(f"File {uploaded_file.name} uploaded successfully!")

##############################################################################
def old_main():
    # Set page config
    st.set_page_config(
        page_title="האוניברסיטה הפתוחה - ייעוץ אקדמי", #config["title"],
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    config, llm_obj = init()
    init_session_state(config, llm_obj)
    init_css()

    # Header
    st.title(config["title"])
    st.subheader(config["description"])

    # Chat history container
    with st.container():
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        st.markdown('</div>', unsafe_allow_html=True)

    # Input area container
    with st.container():
        # Text input row (removed send button column)
        prompt = st.text_input(
            "Message", 
            key="user_input",
            value=st.session_state["input_text"],
            label_visibility="collapsed",
            disabled=st.session_state["processing"]
            #on_change=lambda: handle_input() if st.session_state.user_input else None
        )

        # Bottom buttons row
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("Upload", 
                                         key="upload_button",
                                         type=["pdf"],
                                         label_visibility="collapsed",
                                         disabled=st.session_state["processing"])
        with col2:
            if st.button("🔄", 
                        use_container_width=True,
                        disabled=st.session_state["processing"]):
                st.session_state["messages"] = []
                st.rerun()

        # Handle input when text is entered
        if prompt and not st.session_state["processing"]:  # Only process new input when not already processing
            st.session_state["current_input"] = prompt  # Store the input in a different session state variable
            st.session_state["input_text"] = ""  # Clear the input immediately
            st.session_state["processing"] = True
            st.rerun()
            
        # If we're processing, show the spinner and make the API call
        if st.session_state["processing"] and "current_input" in st.session_state:
            try:
                # Add user message to chat history
                st.session_state["messages"].append({"role": "user", "content": st.session_state["current_input"]})
                
                # Get bot response
                bot_response, client_id = llm_obj.do_query(st.session_state["current_input"], st.session_state["messages"], st.session_state["client_id"])
                
                # Add bot response to chat history
                st.session_state["messages"].append({"role": "assistant", "content": bot_response})
                
            except Exception as e:
                msg = f"Error processing query: Error: {e}"
                print(msg)

            finally:
                # Reset processing state
                st.session_state["processing"] = False
                # Clear the inputs
                st.session_state["input_text"] = ""
                if "current_input" in st.session_state:
                    del st.session_state["current_input"]
                # Rerun to update the UI
                st.rerun()

        # Handle file upload
        if uploaded_file:
            st.success(f"File {uploaded_file.name} uploaded successfully!")

##############################################################################
if __name__ == "__main__":
    main() 
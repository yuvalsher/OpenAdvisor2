from __future__ import annotations
import io
from typing import Literal, TypedDict
import asyncio
import os
from uuid import uuid4

import PyPDF2
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from PydanticAgent import open_university_expert, PydanticAIDeps
from config import all_config

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title=all_config["General"]["title"],
    page_icon=":robot:",
    layout="wide"
)

##############################################################################
def init_session_state():
    # Initialize all session state variables if they don't exist
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "client_id" not in st.session_state:
        st.session_state["client_id"] = str(uuid4())
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

##############################################################################
def init_css():
    st.markdown("""
        <style>
        /* Base RTL styles */
        body, html, p, div, input, label, h1, h2, h3, h4, h5, h6 {
            direction: RTL;
            text-align: right;
        }
        
        /* Container styles */
        [data-testid="stChatMessageContent"] {
            direction: rtl;
            text-align: right;
        }
        
        .stChatFloatingInputContainer {
            direction: rtl;
        }
        
        /* File uploader styles */
        [data-testid="stFileUploader"] {
            direction: rtl;
        }
        
        /* Column layout fixes for RTL */
        [data-testid="column"] {
            direction: rtl;
        }
        </style>
    """, unsafe_allow_html=True)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


##############################################################################
def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          


##############################################################################
async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    try:
        # Prepare dependencies
        deps = PydanticAIDeps(
            supabase=supabase,
            openai_client=openai_client,
            uploaded_files=st.session_state["uploaded_files"]
        )

        # Run the agent in a stream
        async with open_university_expert.run_stream(
            user_input,
            deps=deps,
            message_history=st.session_state.messages[:-1]  # pass entire conversation so far
        ) as result:
            # We'll gather partial text to show incrementally
            partial_text = ""
            message_placeholder = st.empty()

            # Render partial text as it arrives
            async for chunk in result.stream_text(delta=True):
                partial_text += chunk
                message_placeholder.markdown(partial_text)

            # Now that the stream is finished, we have a final result.
            # Add new messages from this run, excluding user-prompt messages
            filtered_messages = [msg for msg in result.new_messages() 
                                if not (hasattr(msg, 'parts') and 
                                        any(part.part_kind == 'user-prompt' for part in msg.parts))]
            st.session_state.messages.extend(filtered_messages)

            # Add the final response to the messages
            st.session_state.messages.append(
                ModelResponse(parts=[TextPart(content=partial_text)])
            )

    except Exception as e:
        error_message = "מצטער, נתקלתי בבעיה בעיבוד הבקשה שלך. אנא נסה שוב."
        st.error(error_message)
        # Add error response to conversation
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=error_message)])
        )
        # Log the error for debugging
        print(f"Error in run_agent_with_streaming: {str(e)}")


##############################################################################
def _read_pdf_content(file: UploadedFile) -> str:
    """Helper method to read PDF content"""
    try:
        pdf_bytes = io.BytesIO(file.getvalue())
        pdf_reader = PyPDF2.PdfReader(pdf_bytes)
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        return text_content
    except Exception as e:
        return None

##############################################################################
async def main():
    init_session_state()
    init_css()
    
    # Initialize clicked state if not exists
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def toggle_clicked():
        st.session_state.clicked = not st.session_state.clicked

    # Header layout with columns
    col1, col2 = st.columns([4, 1], gap="large")
    with col1:
        st.header(all_config["General"]["title"])
    with col2:
        if st.session_state.clicked:
            st.button("סגור קבצים", on_click=toggle_clicked)
        else:
            st.button("העלה קבצים", on_click=toggle_clicked)

    # File upload section
    if st.session_state.clicked:
        uploaded_file = st.file_uploader(
            "העלה קובץ PDF",
            type=["pdf"],
            help="ניתן להעלות קבצים בפורמט PDF בלבד"
        )

        if uploaded_file:
            # Check if file is already processed
            file_already_uploaded = any(
                f["name"] == uploaded_file.name 
                for f in st.session_state["uploaded_files"]
            )
            
            if not file_already_uploaded:
                pdf_content = _read_pdf_content(uploaded_file)
                if pdf_content:
                    st.session_state["uploaded_files"].append({
                        "name": uploaded_file.name, 
                        "content": pdf_content
                    })
                    st.success(f'הקובץ "{uploaded_file.name}" הועלה בהצלחה')
                else:
                    st.error(f"שגיאה בקריאת הקובץ {uploaded_file.name}")

    # Chat container
    messages_container = st.container(border=True, height=600)
    
    with messages_container:
        # Display existing messages
        for msg in st.session_state.messages:
            if isinstance(msg, (ModelRequest, ModelResponse)):
                for part in msg.parts:
                    display_message_part(part)

    # Chat input
    if user_input := st.chat_input(all_config["General"]["Chat_Welcome_Message"]):
        # Append new request to conversation
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # Display user message
        with messages_container:
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Display assistant response
            with st.chat_message("assistant"):
                await run_agent_with_streaming(user_input)

##############################################################################
if __name__ == "__main__":
    asyncio.run(main())

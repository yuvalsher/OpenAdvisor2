from __future__ import annotations
import io
from typing import Literal, TypedDict
import asyncio
import os
from uuid import uuid4
import atexit

import PyPDF2
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI
from bidi.algorithm import get_display
from io import BytesIO

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
        /* Title alignment */
        h2, [data-testid="stMarkdownContainer"] h2 {
            vertical-align: top;
            margin-top: 0 !important;
            padding-top: 0 !important;
            text-align: right;
            display: block;
        }
        .stChatMessage {
            padding: 1rem
        }         
        </style>
    """, unsafe_allow_html=True)

openai_client = None

def get_openai_client():
    global openai_client
    if openai_client is None:
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return openai_client

async def cleanup():
    global openai_client
    if openai_client:
        await openai_client.close()
        openai_client = None

# Register cleanup to run at exit
atexit.register(lambda: asyncio.run(cleanup()))

supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire to suppress warnings (optional)
#logfire.configure(send_to_logfire='never')
logfire.configure(token=os.getenv("LOGFIRE_WRITE_TOKEN"))

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
            openai_client=get_openai_client(),
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
            logfire.info('Agent run completed successfully.', data = partial_text)

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
        st.error(f"Error reading PDF file: {str(e)}")
        return None


##############################################################################
def generate_pdf_from_chat_history(messages: list) -> bytes:
    """
    Generate a PDF from chat history messages.

    The PDF includes a Hebrew title and then displays each conversational turn
    with "שאלה:" for user questions and "תשובה:" for assistant answers.

    This implementation:
      - For ModelRequest (user question): only processes parts with part_kind "user-prompt".
      - For ModelResponse (assistant answer): filters out internal tool call parts.
    
    Dependencies:
      - Markdown: to convert markdown text to HTML.
      - WeasyPrint: to render the HTML (with embedded CSS) into a PDF.
    """
    import markdown  # For converting markdown to HTML
    from weasyprint import HTML  # For generating PDF from HTML

    # Page title (in Hebrew: "Chat History")
    title = "היסטוריית שיחה"

    # CSS styles to enforce RTL layout and basic styling.
    style = """
    <style>
      @page { 
         size: A4; 
         margin: 1cm;
      }
      body { 
         direction: rtl; 
         text-align: right; 
         font-family: "DejaVu Sans", sans-serif; 
         line-height: 1.5;
      }
      h1 { 
         text-align: center; 
         margin-bottom: 1em;
      }
      .question { 
         margin: 10px 0; 
         padding: 10px; 
         border-left: 4px solid #007bff; 
         background-color: #f0f8ff; 
      }
      .answer { 
         margin: 10px 0; 
         padding: 10px; 
         border-right: 4px solid #28a745; 
         background-color: #f8f9fa; 
      }
      .message-title {
         font-weight: bold;
         margin-bottom: 5px;
      }
    </style>
    """

    def get_message_content(part) -> str:
        """
        Retrieve the textual content from a message part.

        This function first attempts to read the 'content' attribute.
        If that returns a list, we join its items into a string.
        If not available (or for alternative types like ToolCallPart),
        we fall back to another attribute (e.g. 'tool_message') or the string representation.
        """
        if hasattr(part, "content"):
            content = part.content
        elif hasattr(part, "tool_message"):
            content = part.tool_message
        else:
            content = str(part)

        # If the content is a list, join the items into a string.
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        return content

    # Build the conversation HTML by iterating over messages.
    content_html = ""
    for msg in messages:
        # Make sure the message has a "parts" attribute.
        if hasattr(msg, "parts"):
            # Import your message types. Adjust the import as needed.
            from pydantic_ai.messages import ModelRequest, ModelResponse

            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    # Only process parts that are actually user prompts.
                    if getattr(part, "part_kind", "") != "user-prompt":
                        continue
                    message_text = get_message_content(part)
                    md_to_html = markdown.markdown(message_text)
                    content_html += (
                        f"<div class='question'>"
                        f"<div class='message-title'>שאלה:</div>"
                        f"{md_to_html}"
                        f"</div>"
                    )
            elif isinstance(msg, ModelResponse):
                for part in msg.parts:
                    # Skip internal tool call parts.
                    if getattr(part, "part_kind", "") == "tool-call":
                        continue
                    message_text = get_message_content(part)
                    md_to_html = markdown.markdown(message_text)
                    content_html += (
                        f"<div class='answer'>"
                        f"<div class='message-title'>תשובה:</div>"
                        f"{md_to_html}"
                        f"</div>"
                    )

    # Combine the CSS and body content into a full HTML document.
    full_html = f"""
    <html>
      <head>
        <meta charset="utf-8">
        {style}
      </head>
      <body>
        <h1>{title}</h1>
        {content_html}
      </body>
    </html>
    """

    # Convert the HTML into PDF bytes using WeasyPrint.
    pdf_bytes = HTML(string=full_html).write_pdf()
    return pdf_bytes


##############################################################################
async def main():
    init_session_state()
    init_css()
    
    # Initialize clicked state if not exists
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def toggle_clicked():
        st.session_state.clicked = not st.session_state.clicked

    # Outer layout: center content column (middle of 3 columns)
    outer_cols = st.columns([1, 4, 1])
    with outer_cols[1]:
        # Header row: create three columns: title, file upload toggle, and save button.
        st.header(all_config["General"]["title"])
        st.markdown(all_config['General']['description'])

    with outer_cols[2]:
        button_row = st.columns([1, 1])
        with button_row[0]:
            button_text = "סגור חזרה" if st.session_state.clicked else "העלה קובץ"
            st.button(button_text, on_click=toggle_clicked)

        with button_row[1]:
            pdf_placeholder = st.empty()
            initial_pdf = generate_pdf_from_chat_history(st.session_state["messages"])
            pdf_placeholder.download_button(
                label="שמור שיחה",
                data=initial_pdf,
                file_name="chat_history.pdf",
                mime="application/pdf",
                key="initial_pdf_download"
            )

    if st.session_state.clicked:
        uploaded_file = st.file_uploader(
            "העלה קובץ PDF",
            type=["pdf"],
            help="ניתן להעלות קבצים בפורמט PDF בלבד"
        )
        if uploaded_file:
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

    st.write("")  # add some vertical spacing below the header row

    # Display currently uploaded files (inside outer central column)
    if st.session_state["uploaded_files"]:
         st.write("קבצים שהועלו:")
         for file in st.session_state["uploaded_files"]:
              st.write(f"- {file['name']}")

    # Chat container (inside outer central column)
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

    # Immediately update the download button placeholder with the latest PDF
    updated_pdf = generate_pdf_from_chat_history(st.session_state["messages"])
    pdf_placeholder.empty()  # clear the previous download button
    pdf_placeholder.download_button(
         label="שמור שיחה",
         data=updated_pdf,
         file_name="chat_history.pdf",
         mime="application/pdf",
         key="updated_pdf_download"
    )

##############################################################################
if __name__ == "__main__":
    asyncio.run(main())

from __future__ import annotations
import io
from fpdf import FPDF  # new import for PDF generation
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
from bidi.algorithm import get_display

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
        st.error(f"Error reading PDF file: {str(e)}")
        return None

##############################################################################
def generate_pdf_from_chat_history(messages) -> bytes:
    """Generate a PDF file containing the entire chat history.
    
    This function uses FPDF to mimic the visual language of the chat.
    It prints a title at the top and then prints messages with labels:
      - User messages (questions) are introduced with a bold "שאלה:" label,
      - Assistant/system messages (answers) are introduced with a bold "תשובה:" label.
    The text is right-to-left (RTL) aligned.
    """
    pdf = FPDF()
    import os
    FONT_FILENAME = "DejaVuSans.ttf"
    FONT_BOLD_FILENAME = "DejaVuSans-Bold.ttf"
    font_path = os.path.join(os.path.dirname(__file__), FONT_FILENAME)
    bold_font_path = os.path.join(os.path.dirname(__file__), FONT_BOLD_FILENAME)
    
    bold_available = False
    if os.path.exists(font_path):
        pdf.add_font('DejaVu', '', font_path, uni=True)
        if os.path.exists(bold_font_path):
            pdf.add_font('DejaVu', 'B', bold_font_path, uni=True)
            bold_available = True
        font_family = 'DejaVu'
    else:
        print(f"Error: {FONT_FILENAME} not found at {font_path}. Hebrew text may not render correctly.")
        font_family = 'Arial'
    
    pdf.add_page()
    
    # ------------------------------------------------------------------------
    # Helper: Sanitize text by removing characters not supported by the font.
    # ------------------------------------------------------------------------
    def sanitize_text(text: str) -> str:
        try:
            # Attempt to locate the actual font key in a case-insensitive way.
            actual_font_key = None
            for key in pdf.fonts.keys():
                if key.lower() == font_family.lower():
                    actual_font_key = key
                    break
            if actual_font_key is None:
                raise KeyError(f"Font {font_family} not found in pdf.fonts.")
    
            cw = pdf.fonts[actual_font_key]["cw"]
            max_idx = len(cw)
            safe_chars = []
            for ch in text:
                if ord(ch) < max_idx:
                    safe_chars.append(ch)
            return "".join(safe_chars)
        except Exception:
            # Fallback: Remove characters with Unicode codepoints > 0xFFFF (commonly problematic emoji)
            return "".join(ch for ch in text if ord(ch) <= 0xFFFF)
    
    # ------------------------------------------------------------------------
    # Helper: Process multi-line text for RTL.
    # ------------------------------------------------------------------------
    def process_rtl_text(text: str) -> str:
        """
        Split text into lines, process each one using get_display with the base direction set to RTL,
        then join the lines in their original order.
        
        This ensures that get_display() will not reverse the order of the lines.
        """
        lines = text.split("\n")
        # Here we pass base_dir='R' so that the reordering is done according to RTL,
        # which prevents get_display() from reversing the line order.
        processed_lines = [get_display(line) for line in lines]
        return "\n".join(processed_lines)
    
    # ------------------------------------------------------------------------
    # Add Title at the top.
    # ------------------------------------------------------------------------
    title_text = all_config["General"].get("title", "Chat History")
    if bold_available:
        pdf.set_font(font_family, 'B', 16)
    else:
        pdf.set_font(font_family, '', 16)
    rtl_title = process_rtl_text(sanitize_text(title_text))
    pdf.multi_cell(0, 10, txt=rtl_title, align='R')
    pdf.ln(10)
    
    # Set the normal text font for subsequent content.
    pdf.set_font(font_family, '', 12)
    
    if not messages:
         default_text = process_rtl_text(sanitize_text("אין היסטוריית שיחה"))
         pdf.multi_cell(0, 10, txt=default_text, align='R')
    
    # ------------------------------------------------------------------------
    # Print each message with a bold label above its content.
    # ------------------------------------------------------------------------
    for msg in messages:
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                # Skip tool-related messages.
                if part.part_kind in ['tool-call', 'tool-return']:
                    continue
                
                if part.part_kind == 'user-prompt':
                    label = "שאלה:"
                    if bold_available:
                        pdf.set_font(font_family, 'B', 12)
                    else:
                        pdf.set_font(font_family, '', 12)
                    label_rtl = process_rtl_text(sanitize_text(label))
                    pdf.multi_cell(0, 10, txt=label_rtl, align='R')
                    
                    pdf.set_font(font_family, '', 12)
                    content = part.content
                    content_rtl = process_rtl_text(sanitize_text(content))
                    pdf.multi_cell(0, 10, txt=content_rtl, align='R')
                
                elif part.part_kind in ['system-prompt', 'text']:
                    label = "תשובה:"
                    if bold_available:
                        pdf.set_font(font_family, 'B', 12)
                    else:
                        pdf.set_font(font_family, '', 12)
                    label_rtl = process_rtl_text(sanitize_text(label))
                    pdf.multi_cell(0, 10, txt=label_rtl, align='R')
                    
                    pdf.set_font(font_family, '', 12)
                    content = part.content
                    content_rtl = process_rtl_text(sanitize_text(content))
                    pdf.multi_cell(0, 10, txt=content_rtl, align='R')
                else:
                    continue
                pdf.ln(1)
            pdf.ln(2)
    
    # ------------------------------------------------------------------------
    # Generate PDF output.
    # ------------------------------------------------------------------------
    try:
        pdf_output = pdf.output(dest="S")
    except UnicodeEncodeError as e:
        st.error("Failed to generate PDF due to Unicode encoding issue. Please ensure DejaVuSans.ttf is available for proper Unicode support.")
        return b""
    
    if isinstance(pdf_output, bytes):
        return pdf_output
    else:
        try:
            pdf_bytes = pdf_output.encode("latin1")
        except Exception:
            pdf_bytes = pdf_output.encode("utf-8")
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

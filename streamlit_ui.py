from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
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

##############################################################################
def init_css():
    # Custom CSS for styling
    st.markdown("""
        <style>
        body, html {
            direction: RTL;
            text-align: right;
        }
        p, div, input, label, h1, h2, h3, h4, h5, h6 {
            direction: RTL;
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
                
        /* File uploader styles 
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
        } */
        </style>
    """, unsafe_allow_html=True)

    # # Add processing class to body if processing
    # if st.session_state["processing"]:
    #     st.markdown("""
    #         <script>
    #             document.body.classList.add('processing');
    #         </script>
    #     """, unsafe_allow_html=True)

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


async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = PydanticAIDeps(
        supabase=supabase,
        openai_client=openai_client
    )

    # Run the agent in a stream
    async with open_university_expert.run_stream(
        user_input,
        deps=deps,
        message_history= st.session_state.messages[:-1],  # pass entire conversation so far
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


async def main():
    st.title(all_config["General"]["title"])
    st.write(all_config["General"]["description"])

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    init_css()
    # Chat input for the user
    user_input = st.chat_input(all_config["General"]["Chat_Welcome_Message"])

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input)


if __name__ == "__main__":
    asyncio.run(main())

from abc import ABC, abstractmethod

class AbstractLlm:

    ##############################################################################
    def __init__(self, faculty_code, config):
        self.faculty_code = faculty_code
        self.config = config

    ##############################################################################
    @abstractmethod
    def init(self):
        pass

    ##############################################################################
    @abstractmethod
    # Prepare and send the query to the LLM.
    # Return the response from the LLM
    def do_query(self, user_input: str, chat_history: list[dict]) -> str:
        pass

    ##############################################################################
    # Prepare and send the query to the LLM.
    # Return the updated chat history
#    def run_query(self, user_input: str) -> list[dict]:
#        chat_history.append({msg_sender: self.user_name, msg_text: user_message})
#        chat_history.append({msg_sender: self.bot_name, msg_text: bot_response})

from abc import ABC, abstractmethod

class AbstractLlm(ABC):

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
    def do_query(self, user_input: str, chat_history: list[dict], client_id: str = None) -> tuple[str, str]:
        """
        Process a query from a client.
        
        Args:
            user_input: The user's query
            chat_history: The chat history
            client_id: The client's unique identifier
            
        Returns:
            tuple: (response_text, client_id)
        """
        pass

    ##############################################################################
    @abstractmethod
    def reset_chat_history(self, client_id: str):
        """Reset chat history for a specific client.
        
        Args:
            client_id: The client's unique identifier
        """
        pass

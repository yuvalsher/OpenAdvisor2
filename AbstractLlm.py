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
    def do_query(self, user_input: str, chat_history: list[dict]) -> str:
        pass

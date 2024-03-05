from intents import Intent
from abc import ABC, abstractmethod


class IIE(ABC):
    @abstractmethod
    def extract(self, prompt: str) -> Intent:
        pass

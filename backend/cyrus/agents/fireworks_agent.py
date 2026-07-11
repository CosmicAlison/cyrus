
import requests
from core.config import settings

class FireworksClient:
    def __init__(self):
        self.api_key = settings.FIREWORKS_KEY
        self.model = "accounts/fireworks/models/minimax-m3"

    def chat(self, messages, max_tokens=4096, temperature=0.2):
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": 40,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "messages": messages
        }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.post(
            settings.FIREWORKS_URL,
            headers=headers,
            json=payload
        )

        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]
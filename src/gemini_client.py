"""A small wrapper around the Gemini API with one job: ask().

This file is copied forward into Weeks 3, 7, 9 and 13.

Two things it adds over calling the SDK directly:
  1. the API key is read in exactly one place
  2. a missing key fails immediately, with a message that says what to do
"""

import os

from dotenv import load_dotenv
from google import genai


class MissingKeyError(RuntimeError):
    """No API key. A setup problem - retrying will not help."""


class GeminiClient:
    """Sends prompts to Gemini and returns text."""

    def __init__(self, model="gemini-2.0-flash"):
        load_dotenv()
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise MissingKeyError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self.model = model
        self.client = genai.Client(api_key=key)

    def ask(self, prompt, system=None):
        """Send one prompt and return the reply as text.

        prompt - the user's question
        system - the assembled context package, from build_prompt().
                 Leave it out for a naive run.
        """
        config = {"system_instruction": system} if system else None
        response = self.client.models.generate_content(
            model=self.model, contents=prompt, config=config
        )
        return (response.text or "").strip()

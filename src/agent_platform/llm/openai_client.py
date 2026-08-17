from openai import AsyncOpenAI

from agent_platform.config import Settings
from agent_platform.llm.base import LLMClient


class OpenAIClient(LLMClient):
    """OpenAI implementation of the LLM client."""

    def __init__(self, settings: Settings) -> None:
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-5-mini"

    async def generate(self, prompt: str) -> str:
        response = await self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        return response.output_text

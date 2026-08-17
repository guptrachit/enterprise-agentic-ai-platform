import asyncio

from agent_platform.config import get_settings
from agent_platform.llm.openai_client import OpenAIClient


async def main() -> None:
    settings = get_settings()
    client = OpenAIClient(settings)

    response = await client.generate(
        "Explain what an LLM is in one sentence."
    )

    print(response)


if __name__ == "__main__":
    asyncio.run(main())

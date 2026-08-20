import asyncio

from agent_platform.config import get_settings
from agent_platform.llm.openai_client import OpenAIClient


async def main() -> None:
    settings = get_settings()
    client = OpenAIClient(settings)

    response = await client.generate("Explain what an LLM is in one sentence.")

    print("Response:", response.text)
    print("Provider:", response.metadata.provider)
    print("Model:", response.metadata.model)
    print("Input tokens:", response.usage.input_tokens)
    print("Output tokens:", response.usage.output_tokens)
    print("Total tokens:", response.usage.total_tokens)
    print("Latency (ms):", round(response.metadata.latency_ms, 2))
    print("Request ID:", response.metadata.request_id)


if __name__ == "__main__":
    asyncio.run(main())

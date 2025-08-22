import json

from llama_cpp import Llama, CreateChatCompletionResponse


class LlamaClient:
    def __init__(self) -> None:
        self._llm = Llama.from_pretrained(
            repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
            filename="*q8_0.gguf",
            chat_format="llama-3",
            verbose=False
        )

    def chat(self, messages: list, tools: list) -> CreateChatCompletionResponse:
        output = self._llm.create_chat_completion(messages, tools=tools, tool_choice="auto")
        return output

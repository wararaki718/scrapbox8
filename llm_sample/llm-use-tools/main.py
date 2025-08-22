import json

from myllm import LlamaClient
from postprocess import postprocess
from tools import tools, available_functions

def main() -> None:
    llm = LlamaClient()

    print("## chat")
    messages = [
        {
            "role": "system",
            "content": "You are Homeboy, a cheerful and helpful home assistant."
        },
        {
            "role": "user",
            "content": "Can you make the room a few degrees warmer?"
        }
    ]
    output = llm.chat(messages, tools)
    result = postprocess(output)
    # print(json.dumps(output, indent=4))
    print(result)
    print()

    print("DONE")


if __name__ == "__main__":
    main()

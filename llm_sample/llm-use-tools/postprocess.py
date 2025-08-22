import json

from tools import available_functions

def postprocess(response: dict) -> list[dict]:
    response_message: dict = response["choices"][0]["message"]
    # print(json.dumps(response["choices"][0], indent=4))
    messages = [response_message]

    if not response_message.get("tool_calls"):
        return messages

    for tool_call in response_message["tool_calls"]:
        function_name = tool_call["function"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call["function"]["arguments"])
        function_response = function_to_call(**function_args)
        messages.append(
            {
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )
    return messages

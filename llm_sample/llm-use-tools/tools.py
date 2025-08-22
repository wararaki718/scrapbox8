from room import get_room_temp, set_room_temp


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_room_temp",
            "description": "Get the room temperature in Fahrenheit.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_room_temp",
            "description": "Set the room temperature in Fahrenheit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "temp": {
                        "type": "integer",
                        "description": "Desired room temperature in Fahrenheit.",
                    },
                },
                "required": ["temp"],
            },
        },
    }
]


available_functions = {
    "get_room_temp": get_room_temp,
    "set_room_temp": set_room_temp,
}

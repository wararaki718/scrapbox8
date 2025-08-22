import random


def get_room_temp() -> str:
    return str(random.randint(60, 80))


def set_room_temp(temp: int) -> str:
    return "DONE"

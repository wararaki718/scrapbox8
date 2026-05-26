import json
import requests


def main() -> None:
    url = "https://dummyjson.com/users"

    response = requests.get(url)
    data = response.json()

    with open("data/raw/users.json", "w") as f:
        json.dump(data, f, indent=2)
    print("saved raw json")
    print("DONE")


if __name__ == "__main__":
    main()

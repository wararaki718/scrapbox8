import pandas as pd


def main() -> None:
    decimals = pd.DataFrame({'TSLA': 2, 'AMZN': 1})
    prices = pd.DataFrame(data={
        'date': ['2021-08-13', '2021-08-07', '2021-08-21'],
        'TSLA': [720.13, 716.22, 731.22],
        'AMZN': [3316.50, 3200.50, 3100.23],
    })
    rounded_prices = prices.round(decimals=decimals)
    print(rounded_prices)
    print("DONE")


if __name__ == "__main__":
    main()

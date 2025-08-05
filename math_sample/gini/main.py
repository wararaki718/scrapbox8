import numpy as np
from gini import gini, gini2, gini3, gini4


def main() -> None:
    X = np.array([
        311590, 234758, 212149, 196732, 141475,
    ])

    print("gini:", gini(X))
    print("gini2:", gini2(X))
    print("gini3:", gini3(X))
    print("gini4:", gini4(X))
    print("DONE")


if __name__ == "__main__":
    main()

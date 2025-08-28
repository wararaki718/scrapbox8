from scipy import optimize, sparse, special, linalg


def main() -> None:
    print("accurate type hints:")
    x = special.factorial(99)
    print(x)

    y = special.factorial(99, exact=True)
    print(y)

    result = optimize.minimize(lambda x: x**2, 1)
    min_x = result.x
    print(min_x)

    min_f = result.fun
    print(min_f)

    sp1d = sparse.coo_array([0, 1])
    print(sp1d)

    sp2d = sparse.coo_array([[0, 1], [2, 3]])
    print(sp2d)
    print()

    print("precise type hints:")
    # x = special.factorial(1.5, exact=True)
    # print(x)

    z = linalg.sqrtm([[1, 0], [0, 1]], disp=True)
    print(z)

    a = linalg.toeplitz([[1, 2], [1, 3]])
    print(a)

    print("DONE")


if __name__ == "__main__":
    main()

import torch


def main() -> None:
    data = torch.randn((5, 3))
    corr_matrix = torch.corrcoef(data.T)
    print("Correlation Matrix:")
    print(corr_matrix)
    print("--------------")

    centerd = data - data.mean(dim=0)
    cov_matrix = centerd.T @ centerd / (data.size(0) - 1)
    print("Covariance Matrix:")
    print(cov_matrix)
    print("--------------")

    print(cov_matrix.diag())
    tmp = torch.pow(1 - cov_matrix.diag(), 2)
    print(tmp)
    print("DONE")


if __name__ == "__main__":
    main()

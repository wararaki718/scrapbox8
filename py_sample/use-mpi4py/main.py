from mpi4py import MPI


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        data = {'a': 7, 'b': 3.14}
        comm.send(data, dest=1, tag=11)
        print("send")
    elif rank == 1:
        data = comm.recv(source=0, tag=11)
        print("recieve: ", data)

    print("DONE")


if __name__ == "__main__":
    main()

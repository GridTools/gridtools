#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    printf("Total MPI ranks: %d\n", world_size);

    MPI_Finalize();
    return 0;
}

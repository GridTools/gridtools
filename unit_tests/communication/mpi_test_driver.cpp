#include <mpi.h>
#include "./check_flags.hpp"
#include <iostream>
#include <fstream>
#include <cuda.h>
#include "gtest/gtest.h"

#include "./mpi_listener.hpp"

/* device_binding added by Devendar Bureddy, OSU */
void device_binding() {

    int local_rank = 0 /*, num_local_procs*/;
    int dev_count, use_dev_count, my_dev_id;
    char *str;

    printf("HOME %s\n", getenv("HOME"));

    if ((str = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL) {
        local_rank = atoi(str);
        printf("MV2_COMM_WORLD_LOCAL_RANK %s\n", str);
    } else if ((str = getenv("SLURM_LOCALID")) != NULL) {
        local_rank = atoi(str);
        printf("SLURM_LOCALID %s\n", str);
    }

    if ((str = getenv("MPISPAWN_LOCAL_NPROCS")) != NULL) {
        // num_local_procs = atoi (str);
        printf("MPISPAWN_LOCAL_NPROCS %s\n", str);
    }

    cudaGetDeviceCount(&dev_count);
    if ((str = getenv("NUM_GPU_DEVICES")) != NULL) {
        use_dev_count = atoi(str);
        printf("NUM_GPU_DEVICES %s\n", str);
    } else {
        use_dev_count = dev_count;
        printf("NUM_GPU_DEVICES %d\n", use_dev_count);
    }

    my_dev_id = local_rank % use_dev_count;
    printf("local rank = %d dev id = %d\n", local_rank, my_dev_id);
    cudaSetDevice(my_dev_id);
}

int main(int argc, char **argv) {

    device_binding();

    // We need to set the communicator policy at the top level
    // this allows us to build multiple communicators in the tests
    MPI_Init(&argc, &argv);
    gridtools::GCL_Init(argc, argv);

    // initialize google test environment
    testing::InitGoogleTest(&argc, argv);

    // set up a custom listener that prints messages in an MPI-friendly way
    auto& listeners = testing::UnitTest::GetInstance()->listeners();
    // first delete the original printer
    delete listeners.Release(listeners.default_result_printer());
    // now add our custom printer
    listeners.Append(new mpi_listener("results_global_communication"));

    // record the local return value for tests run on this mpi rank
    //      0 : success
    //      1 : failure
    auto result = RUN_ALL_TESTS();
    decltype(result) global_result{};

    MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Finalize();
    // perform global collective, to ensure that all ranks return
    // the same exit code
    return global_result;
}

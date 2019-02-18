/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "./check_flags.hpp"
#include "gtest/gtest.h"
#include <fstream>
#include <iostream>
#include <mpi.h>

#include "../../communication/GCL.hpp"
#include "./device_binding.hpp"
#include "./mpi_listener.hpp"

int main(int argc, char **argv) {

    device_binding();

    // We need to set the communicator policy at the top level
    // this allows us to build multiple communicators in the tests
    MPI_Init(&argc, &argv);
    gridtools::GCL_Init(argc, argv);

    // initialize google test environment
    testing::InitGoogleTest(&argc, argv);

    // set up a custom listener that prints messages in an MPI-friendly way
    auto &listeners = testing::UnitTest::GetInstance()->listeners();
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

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    // perform global collective, to ensure that all ranks return
    // the same exit code
    return global_result;
}

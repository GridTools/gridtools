/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#include <mpi.h>
#include "./check_flags.hpp"
#include <iostream>
#include <fstream>
#ifdef __CUDACC__
#include <cuda.h>
#endif
#include "gtest/gtest.h"

#include "./mpi_listener.hpp"
#include "./device_binding.hpp"

int main(int argc, char **argv) {

#ifdef _USE_GPU_
    device_binding();
#endif

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

    MPI_Finalize();
    // perform global collective, to ensure that all ranks return
    // the same exit code
    return global_result;
}

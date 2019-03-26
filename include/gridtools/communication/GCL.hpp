/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <iostream>
#ifdef GCL_MPI
#include <mpi.h>
#endif

#ifdef GCL_GPU
#include "../common/cuda_util.hpp"
#endif
#include "../common/host_device.hpp"

#ifdef GCL_TRACE
#include "high_level/stats_collector.hpp"
#endif

#include "low_level/gcl_arch.hpp"

#ifdef GCL_GPU
#ifdef GCL_MULTI_STREAMS
#ifdef GCL_USE_3
extern cudaStream_t ZL_stream;
extern cudaStream_t &ZU_stream;
extern cudaStream_t YL_stream;
extern cudaStream_t &YU_stream;
extern cudaStream_t XL_stream;
extern cudaStream_t &XU_stream;
#else
extern cudaStream_t ZL_stream;
extern cudaStream_t ZU_stream;
extern cudaStream_t YL_stream;
extern cudaStream_t YU_stream;
extern cudaStream_t XL_stream;
extern cudaStream_t XU_stream;
#endif
#else
#define ZL_stream 0
#define ZU_stream 0
#define YL_stream 0
#define YU_stream 0
#define XL_stream 0
#define XU_stream 0
#endif
#endif

namespace gridtools {

#ifdef GCL_MPI
    extern MPI_Comm GCL_WORLD;
#else
    extern int GCL_WORLD;
#endif
    extern int PID;
    extern int PROCS;

    void GCL_Init(int argc, char **argv);

    void GCL_Init();

    void GCL_Finalize();

} // namespace gridtools

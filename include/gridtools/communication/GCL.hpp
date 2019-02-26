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

#include "../common/host_device.hpp"

#ifdef GCL_TRACE
#include "high-level/stats_collector.hpp"
#endif

#include "low-level/gcl_arch.hpp"

#ifdef GCL_GPU

// workaround that uses host buffering to avoid bad sends for messages larger than 512 kB on Cray systems
//#define GCL_HOSTWORKAROUND

inline bool checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cout << cudaGetErrorString(status) << std::endl;
        return false;
    } else
        return true;
}
#endif

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

    enum packing_version { version_mpi_pack = 0, version_datatype, version_manual };

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

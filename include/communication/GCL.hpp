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
#ifndef _GCL_H_
#define _GCL_H_

#include <iostream>
#ifdef _GCL_MPI_
#include <mpi.h>
#endif

#include "../common/host_device.hpp"

#ifdef GCL_TRACE
#include "high-level/stats_collector.hpp"
#endif

#include "low-level/gcl_arch.hpp"

#ifdef _GCL_GPU_

// workaround that uses host buffering to avoid bad sends for messages larger than 512 kB on Cray systems
//#define HOSTWORKAROUND

#define _USE_DATATYPES_

inline bool checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cout << cudaGetErrorString(status) << std::endl;
        return false;
    } else
        return true;
}
#endif

#ifdef _GCL_GPU_
#ifdef GCL_MULTI_STREAMS
#ifdef GCL_USE_3
#ifndef SUPPRESS_MESSAGES
#pragma message "Using 3 streams for packing and unpaking in GCL"
#endif
extern cudaStream_t ZL_stream;
extern cudaStream_t &ZU_stream;
extern cudaStream_t YL_stream;
extern cudaStream_t &YU_stream;
extern cudaStream_t XL_stream;
extern cudaStream_t &XU_stream;
#else
#ifndef SUPPRESS_MESSAGES
#pragma message "Using 6 streams for packing and unpaking in GCL"
#endif
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

#ifdef _GCL_MPI_
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

#endif

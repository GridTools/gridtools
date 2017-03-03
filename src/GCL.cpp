/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include "communication/GCL.hpp"

#ifdef _GCL_GPU_
#ifdef GCL_MULTI_STREAMS
#ifdef GCL_USE_3
cudaStream_t ZL_stream;
cudaStream_t &ZU_stream = ZL_stream;
cudaStream_t YL_stream;
cudaStream_t &YU_stream = YL_stream;
cudaStream_t XL_stream;
cudaStream_t &XU_stream = XL_stream;
#else
cudaStream_t ZL_stream;
cudaStream_t ZU_stream;
cudaStream_t YL_stream;
cudaStream_t YU_stream;
cudaStream_t XL_stream;
cudaStream_t XU_stream;
#endif
#endif
#endif

namespace gridtools {
#ifdef _GCL_MPI_
    MPI_Comm GCL_WORLD;
    int PID;
    int PROCS;

    namespace _impl {
        void GCL_Real_Init(int argc, char **argv) {
            int ready;
            MPI_Initialized(&ready);
            if (!ready)
                MPI_Init(&argc, &argv);

            GCL_WORLD = MPI_COMM_WORLD;
            MPI_Comm_rank(GCL_WORLD, &PID);
            MPI_Comm_size(GCL_WORLD, &PROCS);

#ifdef GCL_MULTI_STREAMS
#ifdef GCL_USE_3
            cudaStreamCreate(&ZL_stream);
            cudaStreamCreate(&YL_stream);
            cudaStreamCreate(&XL_stream);
#else
            cudaStreamCreate(&ZL_stream);
            cudaStreamCreate(&ZU_stream);
            cudaStreamCreate(&YL_stream);
            cudaStreamCreate(&YU_stream);
            cudaStreamCreate(&XL_stream);
            cudaStreamCreate(&XU_stream);
#endif
#endif
        }
    }

    void GCL_Init(int argc, char **argv) { _impl::GCL_Real_Init(argc, argv); }

    void GCL_Init() {
        int arg = 1;
        _impl::GCL_Real_Init(arg, 0);
    }

    void GCL_Finalize() {
#ifdef GCL_MULTI_STREAMS
#ifdef GCL_USE_3
        cudaStreamDestroy(ZL_stream);
        cudaStreamDestroy(YL_stream);
        cudaStreamDestroy(XL_stream);
#else
        cudaStreamDestroy(ZL_stream);
        cudaStreamDestroy(ZU_stream);
        cudaStreamDestroy(YL_stream);
        cudaStreamDestroy(YU_stream);
        cudaStreamDestroy(XL_stream);
        cudaStreamDestroy(XU_stream);
#endif
#endif
        MPI_Finalize();
    }

#ifdef GCL_TRACE
    // initialize static instance_ to NULL
    template <>
    stats_collector< 1 > *stats_collector< 1 >::instance_ = 0;
    template <>
    stats_collector< 2 > *stats_collector< 2 >::instance_ = 0;
    template <>
    stats_collector< 3 > *stats_collector< 3 >::instance_ = 0;

    // convenient handles for the singleton instances for 2D and 3D grids
    stats_collector< 3 > &stats_collector_3D = *stats_collector< 3 >::instance();
    stats_collector< 2 > &stats_collector_2D = *stats_collector< 2 >::instance();
#endif
#else
    int GCL_WORLD;
    int PID;
    int PROCS;

    void GCL_Init(int argc, char **argv) {
        PROCS = 1;
        PID = 0;
    }

    void GCL_Init() {
        PROCS = 1;
        PID = 0;
    }

    void GCL_Finalize() {}
#endif

} // namespace gridtools

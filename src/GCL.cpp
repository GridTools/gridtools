/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/communication/GCL.hpp>

#ifdef GCL_GPU
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
#ifdef GCL_MPI
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
            GT_CUDA_CHECK(cudaStreamCreate(&ZL_stream));
            GT_CUDA_CHECK(cudaStreamCreate(&YL_stream));
            GT_CUDA_CHECK(cudaStreamCreate(&XL_stream));
#else
            GT_CUDA_CHECK(cudaStreamCreate(&ZL_stream));
            GT_CUDA_CHECK(cudaStreamCreate(&ZU_stream));
            GT_CUDA_CHECK(cudaStreamCreate(&YL_stream));
            GT_CUDA_CHECK(cudaStreamCreate(&YU_stream));
            GT_CUDA_CHECK(cudaStreamCreate(&XL_stream));
            GT_CUDA_CHECK(cudaStreamCreate(&XU_stream));
#endif
#endif
        }
    } // namespace _impl

    void GCL_Init(int argc, char **argv) { _impl::GCL_Real_Init(argc, argv); }

    void GCL_Init() {
        int arg = 1;
        _impl::GCL_Real_Init(arg, 0);
    }

    void GCL_Finalize() {
#ifdef GCL_MULTI_STREAMS
#ifdef GCL_USE_3
        GT_CUDA_CHECK(cudaStreamDestroy(ZL_stream));
        GT_CUDA_CHECK(cudaStreamDestroy(YL_stream));
        GT_CUDA_CHECK(cudaStreamDestroy(XL_stream));
#else
        GT_CUDA_CHECK(cudaStreamDestroy(ZL_stream));
        GT_CUDA_CHECK(cudaStreamDestroy(ZU_stream));
        GT_CUDA_CHECK(cudaStreamDestroy(YL_stream));
        GT_CUDA_CHECK(cudaStreamDestroy(YU_stream));
        GT_CUDA_CHECK(cudaStreamDestroy(XL_stream));
        GT_CUDA_CHECK(cudaStreamDestroy(XU_stream));
#endif
#endif
        MPI_Finalize();
    }

#ifdef GCL_TRACE
    // initialize static instance_ to nullptr
    template <>
    stats_collector<1> *stats_collector<1>::instance_ = 0;
    template <>
    stats_collector<2> *stats_collector<2>::instance_ = 0;
    template <>
    stats_collector<3> *stats_collector<3>::instance_ = 0;

    // convenient handles for the singleton instances for 2D and 3D grids
    stats_collector<3> &stats_collector_3D = *stats_collector<3>::instance();
    stats_collector<2> &stats_collector_2D = *stats_collector<2>::instance();
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

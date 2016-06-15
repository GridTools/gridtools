/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "communication/GCL.hpp"

#ifdef _GCL_GPU_
#ifdef GCL_MULTI_STREAMS
#ifdef GCL_USE_3
cudaStream_t ZL_stream ;
cudaStream_t& ZU_stream = ZL_stream;
cudaStream_t YL_stream ;
cudaStream_t& YU_stream = YL_stream;
cudaStream_t XL_stream ;
cudaStream_t& XU_stream = XL_stream;
#else
cudaStream_t ZL_stream ;
cudaStream_t ZU_stream ;
cudaStream_t YL_stream ;
cudaStream_t YU_stream ;
cudaStream_t XL_stream ;
cudaStream_t XU_stream ;
#endif
#endif
#endif

namespace gridtools {
#ifdef _GCL_MPI_
    MPI_Comm GCL_WORLD;
    int PID;
    int PROCS;

    namespace _impl {
        void GCL_Real_Init(int argc, char** argv) {
            int ready;
            MPI_Initialized(&ready);
            if (!ready)
                MPI_Init(&argc, &argv);

            GCL_WORLD = MPI_COMM_WORLD;
            MPI_Comm_rank(GCL_WORLD, &PID);
            MPI_Comm_size(GCL_WORLD, &PROCS);

#ifdef GCL_MULTI_STREAMS
#ifdef GCL_USE_3
            cudaStreamCreate ( &ZL_stream );
            cudaStreamCreate ( &YL_stream );
            cudaStreamCreate ( &XL_stream );
#else
            cudaStreamCreate ( &ZL_stream );
            cudaStreamCreate ( &ZU_stream );
            cudaStreamCreate ( &YL_stream );
            cudaStreamCreate ( &YU_stream );
            cudaStreamCreate ( &XL_stream );
            cudaStreamCreate ( &XU_stream );
#endif
#endif
        }
    }

    void GCL_Init(int argc, char** argv) {
        _impl::GCL_Real_Init(argc, argv);
    }

    void GCL_Init() {
        int ready;
        int arg=1;
        _impl::GCL_Real_Init(arg, 0);
    }

    void GCL_Finalize() {
#ifdef GCL_MULTI_STREAMS
#ifdef GCL_USE_3
        cudaStreamDestroy ( ZL_stream );
        cudaStreamDestroy ( YL_stream );
        cudaStreamDestroy ( XL_stream );
#else
        cudaStreamDestroy ( ZL_stream );
        cudaStreamDestroy ( ZU_stream );
        cudaStreamDestroy ( YL_stream );
        cudaStreamDestroy ( YU_stream );
        cudaStreamDestroy ( XL_stream );
        cudaStreamDestroy ( XU_stream );
#endif
#endif
        MPI_Finalize();
    }

#ifdef GCL_TRACE
    // initialize static instance_ to NULL
    template<> stats_collector<1>* stats_collector<1>::instance_ = 0;
    template<> stats_collector<2>* stats_collector<2>::instance_ = 0;
    template<> stats_collector<3>* stats_collector<3>::instance_ = 0;

    // convenient handles for the singleton instances for 2D and 3D grids
    stats_collector<3> &stats_collector_3D = *stats_collector<3>::instance();
    stats_collector<2> &stats_collector_2D = *stats_collector<2>::instance();
#endif
#else
    int GCL_WORLD;
    int PID;
    int PROCS;

    void GCL_Init(int argc, char** argv) {
        PROCS = 1;
        PID = 0;
    }

    void GCL_Init() {
        PROCS = 1;
        PID = 0;
    }

    void GCL_Finalize() {  }
#endif


} // namespace gridtools

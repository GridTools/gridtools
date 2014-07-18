
/*
Copyright (c) 2012, MAURO BIANCO, UGO VARETTO, SWISS NATIONAL SUPERCOMPUTING CENTRE (CSCS)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Swiss National Supercomputing Centre (CSCS) nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL MAURO BIANCO, UGO VARETTO, OR 
SWISS NATIONAL SUPERCOMPUTING CENTRE (CSCS), BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef _GCL_H_
#define _GCL_H_

#include <iostream>
#ifdef _GCL_MPI_
#include <mpi.h>
#endif

#ifdef _GCL_GPU_
#include <cuda_runtime.h>
#else
#define __host__
#define __device__

#endif

#ifdef GCL_TRACE
#include <stats_collector.h>
#endif


#include <utils/boollist.h>
#include <gcl_arch.h>


#ifdef _GCL_GPU_

// workaround that uses host buffering to avoid bad sends for messages larger than 512 kB on Cray systems
//#define HOSTWORKAROUND

#define _USE_DATATYPES_


inline bool checkCudaStatus( cudaError_t status ) {
  if( status != cudaSuccess ) {
    std::cout << cudaGetErrorString( status ) << std::endl;
    return false;
  } else return true;
}
#endif

#ifdef _GCL_GPU_
#ifdef GCL_MULTI_STREAMS
#ifdef GCL_USE_3
#pragma message "Using 3 streams for packing and unpaking in GCL"
    extern cudaStream_t ZL_stream ;
    extern cudaStream_t& ZU_stream ;
    extern cudaStream_t YL_stream ;
    extern cudaStream_t& YU_stream ;
    extern cudaStream_t XL_stream ;
    extern cudaStream_t& XU_stream ;
#else
#pragma message "Using 6 streams for packing and unpaking in GCL"
    extern cudaStream_t ZL_stream ;
    extern cudaStream_t ZU_stream ;
    extern cudaStream_t YL_stream ;
    extern cudaStream_t YU_stream ;
    extern cudaStream_t XL_stream ;
    extern cudaStream_t XU_stream ;
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

namespace GCL {

enum packing_version {version_mpi_pack=0, version_datatype, version_manual};

#ifdef _GCL_MPI_
  extern MPI_Comm GCL_WORLD;
#else
  extern int GCL_WORLD;
#endif
  extern int PID;
  extern int PROCS;

  void GCL_Init(int argc, char** argv);

  void GCL_Init();

  void GCL_Finalize();

} // namespace GCL


#endif

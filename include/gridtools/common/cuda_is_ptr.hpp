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

#pragma once

#include "../common/defs.hpp"
#include "../common/host_device.hpp"
#include <iostream>

namespace gridtools {
    /**
     * @brief returns true if ptr is pointing to CUDA device memory
     * @warning A ptr which is not allocated by cudaMalloc, cudaMallocHost, ... (a normal cpu ptr) will emit an error in
     * cuda-memcheck.
     */
    GT_FUNCTION_HOST bool is_gpu_ptr(void *ptr) {
#ifndef _USE_GPU_
        return false;
#else
        cudaPointerAttributes ptrAttributes;
        cudaError_t error = cudaPointerGetAttributes(&ptrAttributes, ptr);
        if (error == cudaSuccess)
            return ptrAttributes.memoryType == cudaMemoryTypeDevice;
        else if (error == cudaErrorInvalidValue) {
            cudaGetLastError(); // clear the error code
            return false;       // it is not a ptr allocated with cudaMalloc, cudaMallocHost, ...
        } else {
            std::cerr << "CUDA ERROR in cudaPointerGetAttributes(): " << cudaGetErrorString(error) << std::endl;
        }
        return false;
#endif
    }
}

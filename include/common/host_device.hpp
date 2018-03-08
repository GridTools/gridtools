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
/**
@file
@brief definition of macros for host/GPU
*/
#ifdef _USE_GPU_
#include <cuda_runtime.h>
#endif

#ifdef __GNUC__
#define GT_FORCE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define GT_FORCE_INLINE inline __forceinline
#else
#define GT_FORCE_INLINE inline
#endif

#ifndef GT_FUNCTION
#ifdef __CUDACC__
#define GT_FUNCTION __host__ __device__ __forceinline__
#define GT_FUNCTION_HOST __host__ __forceinline__
#define GT_FUNCTION_DEVICE __device__ __forceinline__
#define GT_FUNCTION_WARNING __host__ __device__ __forceinline__
#else
#define GT_FUNCTION GT_FORCE_INLINE
#define GT_FUNCTION_HOST GT_FORCE_INLINE
#define GT_FUNCTION_DEVICE GT_FORCE_INLINE
#define GT_FUNCTION_WARNING GT_FORCE_INLINE
#endif
#endif

#ifndef GT_KERNEL
#ifdef __CUDACC__
#define GT_KERNEL __global__
#else
#define GT_KERNEL
#endif
#endif

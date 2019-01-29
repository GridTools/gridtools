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
/** \ingroup common
    @{
    \defgroup hostdevice Host-Device Macros
    @{
*/

#ifdef _USE_GPU_
#include <cuda_runtime.h>
#endif

#if defined(__CUDACC__)
#define GT_FORCE_INLINE __forceinline__
#elif defined(__GNUC__)
#define GT_FORCE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define GT_FORCE_INLINE inline __forceinline
#else
#define GT_FORCE_INLINE inline
#endif

/**
 * @def GT_FUNCTION
 * Function attribute macro to be used for host-device functions.
 */
/**
 * @def GT_FUNCTION_HOST
 * Function attribute macro to be used for host-only functions.
 */
/**
 * @def GT_FUNCTION_DEVICE
 * Function attribute macro to be used for device-only functions.
 */
/**
 * @def GT_FUNCTION_WARNING
 * Function attribute macro to be used for host-only functions that might call a host-device
 * function. This macro is only needed to supress NVCC warnings.
 */

#ifdef __CUDACC__
#define GT_HOST_DEVICE __host__ __device__
#define GT_DEVICE __device__
#define GT_HOST __host__
#else
#define GT_HOST_DEVICE
#define GT_DEVICE
#define GT_HOST
#endif

#ifndef GT_FUNCTION
#define GT_FUNCTION GT_HOST_DEVICE GT_FORCE_INLINE
#endif

#ifndef GT_FUNCTION_WARNING
#define GT_FUNCTION_WARNING GT_HOST_DEVICE GT_FORCE_INLINE
#endif

#ifndef GT_FUNCTION_HOST
#define GT_FUNCTION_HOST GT_HOST GT_FORCE_INLINE
#endif

#ifndef GT_FUNCTION_DEVICE
#define GT_FUNCTION_DEVICE GT_DEVICE GT_FORCE_INLINE
#endif

/**
 *   A helper to implement a family of functions which are different from each other only by target specifies.
 *
 *   It uses the same design pattern as `BOOST_PP_ITERATE` does.
 *   For example if one wants to define a function with any possible combination of `__host__` and `__device__`
 *   specifiers he needs to write the following code:
 *
 *   foo.hpp:
 *
 *   \code
 *   // here we query if this file is used in the context of iteration
 *   #ifndef GT_TARGET_ITERATING
 *
 *   // note that you can't use `#pragma once` here and have to use classic header guards instead
 *   #ifndef FOO_HPP_
 *   #define FOO_HPP_
 *
 *   #include <path/to/this/file/host_device.hpp>
 *
 *   // we need to provide GT_ITERATE_ON_TARGETS() with the name of the current file to include it back during
 *   // iteration process. GT_FILENAME is a hardcoded name that GT_ITERATE_ON_TARGETS() will use.
 *   #define GT_FILENAME <path/to/the/user/file/foo.hpp>
 *
 *   // iteration takes place here
 *   #include GT_ITERATE_ON_TARGETS()
 *
 *   // cleanup
 *   #undef GT_FILENAME
 *
 *   #endif
 *   #else
 *
 *   // here is the code that will be included several times during the iteration process
 *
 *   namespace my {
 *     // GT_TARGET_NAMESPACE will be defined by GT_ITERATE_ON_TARGETS() for you.
 *     // It could be either `namespace host` or `namespace device` or `namespace host_device`.
 *     // one of those namespaces would be defined as `inline`
 *     GT_TARGET_NAMESPACE {
 *        // GT_TARGET will be defined by GT_ITERATE_ON_TARGETS() for you.
 *        // It will contain target specifier that is needed it the given context.
 *        GT_TARGET void foo() {}
 *     }
 *   }
 *
 *   #endif
 *   \endcode
 *
 *   By including "file.hpp" file the following symbols would be available:
 *   - `my::foo`
 *   - `my::host::foo`
 *   - `my::device::foo`
 *   - `my::host_device::foo`
 *
 *   where:
 *
 *   - `my::host::foo` has no specifiers.
 *   - `my::foo` is resolved to `my::host::foo`
 *
 *   If compiling with CUDA, `my::device::foo` has `__device__` specifier, `my::host_device::foo` has
 *   `__host__ __device__` specifier.
 *
 *   Otherwise `my::device::foo` and `my::host_device::foo` are resolved to `my::host::foo`.
 */
#define GT_ITERATE_ON_TARGETS() <gridtools/common/iterate_on_host_device.hpp>

/** @} */
/** @} */

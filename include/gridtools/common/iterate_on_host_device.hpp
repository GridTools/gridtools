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

// DON'T USE #pragma once HERE!!!

#if !defined(GT_FILENAME)
#error GT_FILENAME is not defined
#endif

#if defined(GT_TARGET_ITERATING)
#error nesting target iterating is not supported
#endif

#if defined(GT_TARGET)
#error GT_TARGET should not be defined outside of this file
#endif

#if defined(GT_TARGET_NAMESPACE)
#error GT_TARGET_NAMESPACE should not be defined outside of this file
#endif

#define GT_TARGET_ITERATING

#ifdef __CUDACC__

#define GT_TARGET_NAMESPACE inline namespace host_device
#define GT_TARGET GT_HOST_DEVICE
#include GT_FILENAME
#undef GT_TARGET
#undef GT_TARGET_NAMESPACE

#define GT_TARGET_NAMESPACE namespace host
#define GT_TARGET GT_HOST
#include GT_FILENAME
#undef GT_TARGET
#undef GT_TARGET_NAMESPACE

#define GT_TARGET_NAMESPACE namespace device
#define GT_TARGET GT_DEVICE
#include GT_FILENAME
#undef GT_TARGET
#undef GT_TARGET_NAMESPACE

#else

#define GT_TARGET_NAMESPACE   \
    inline namespace host {}  \
    namespace device {        \
        using namespace host; \
    }                         \
    namespace host_device {   \
        using namespace host; \
    }                         \
    inline namespace host
#define GT_TARGET GT_HOST
#include GT_FILENAME
#undef GT_TARGET
#undef GT_TARGET_NAMESPACE

#endif

#undef GT_TARGET_ITERATING

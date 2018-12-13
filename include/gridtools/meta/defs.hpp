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

// some compilers have the problem that template alias instantiations have exponential complexity
#if !defined(GT_BROKEN_TEMPLATE_ALIASES)
#if defined(__CUDACC_VER_MAJOR__)
// CUDA 9.0 and 9.1 have an different problem (not related to the exponential complexity of template alias
// instantiation) see https://github.com/eth-cscs/gridtools/issues/976
#define GT_BROKEN_TEMPLATE_ALIASES (__CUDACC_VER_MAJOR__ < 9)
#elif defined(__INTEL_COMPILER)
#define GT_BROKEN_TEMPLATE_ALIASES (__INTEL_COMPILER < 1800)
#elif defined(__clang__)
#define GT_BROKEN_TEMPLATE_ALIASES 0
#elif defined(__GNUC__) && defined(__GNUC_MINOR__)
#define GT_BROKEN_TEMPLATE_ALIASES (__GNUC__ * 10 + __GNUC_MINOR__ < 47)
#else
#define GT_BROKEN_TEMPLATE_ALIASES 0
#endif
#endif

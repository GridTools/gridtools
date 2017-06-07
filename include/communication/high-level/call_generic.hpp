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
#if !BOOST_PP_IS_ITERATING

#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

// clang-format off
#define BOOST_PP_ITERATION_PARAMS_1 (3, (1, GCL_MAX_FIELDS, <communication/high-level/call_generic.hpp>))
// clang-format on
#include BOOST_PP_ITERATE()

#else

#define noi BOOST_PP_ITERATION()

#define _PACK_F_NAME(x) m_pack##x##_generic_nv
#define PACK_F_NAME(x) _PACK_F_NAME(x)

#define _PACK_FILE_NAME(x) invoke_kernels_##x##_PP.hpp
#define PACK_FILE_NAME(x) _PACK_FILE_NAME(x)

#define _print_FIELDS(z, m, s) \
    (*filep) << "fieldx " << field##m << "\n" << sizeof(typename FOTF_T##m::value_type) << std::endl;
#define print_FIELDS(m) BOOST_PP_REPEAT(m, _print_FIELDS, nil)

template < BOOST_PP_ENUM_PARAMS(noi, typename FOTF_T) >
void PACK_F_NAME(KERNEL_TYPE)(
    BOOST_PP_ENUM_BINARY_PARAMS(noi, FOTF_T, const &field), void **d_msgbufTab, const int *d_msgsize) {
// print_FIELDS(noi);

#define QUOTE(x) #x
#define _QUOTE(x) QUOTE(x)
#include _QUOTE(PACK_FILE_NAME(KERNEL_TYPE))
#undef QUOTE
#undef _QUOTE
}

#define _UNPACK_F_NAME(x) m_unpack##x##_generic_nv
#define UNPACK_F_NAME(x) _UNPACK_F_NAME(x)

#define _UNPACK_FILE_NAME(x) invoke_kernels_U_##x##_PP.hpp
#define UNPACK_FILE_NAME(x) _UNPACK_FILE_NAME(x)

#define MSTR(x) #x

template < BOOST_PP_ENUM_PARAMS(noi, typename FOTF_T) >
void UNPACK_F_NAME(KERNEL_TYPE)(
    BOOST_PP_ENUM_BINARY_PARAMS(noi, FOTF_T, const &field), void **d_msgbufTab_r, int *d_msgsize_r) {

#define QUOTE(x) #x
#define _QUOTE(x) QUOTE(x)
#include _QUOTE(UNPACK_FILE_NAME(KERNEL_TYPE))
#undef QUOTE
#undef _QUOTE
}

#undef PACK_F_NAME
#undef _PACK_F_NAME
#undef PACK_FILE_NAME
#undef _PACK_FILE_NAME
#undef QUOTE_PACK_FILE_NAME
#undef UNPACK_F_NAME
#undef _UNPACK_F_NAME
#undef UNPACK_FILE_NAME
#undef _UNPACK_FILE_NAME
#undef QUOTE_UNPACK_FILE_NAME
#undef print_FIELDS
#undef _print_FIELDS
#undef noi

#endif

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

#include <gridtools/stencil-composition/sid/concept.cpp>

#include <gtest/gtest.h>

namespace gridtools {
    namespace {
        static_assert(!is_sid<void, 3>{}, "");

        struct a_ref {};
        struct a_ptr {};
        struct strides {};
        a_ref sid_deref(a_ptr, strides, ...);
        void sid_shift(a_ptr &, strides, ...);
        struct a_sid {
            friend strides sid_get_strides(a_sid);
            friend a_ptr sid_get_origin(a_sid);
        };

        static_assert(is_sid<a_sid, 3, 4>{}, "");
        static_assert(std::is_same<sid::ptr_type<a_sid>, a_ptr>{}, "");
        static_assert(std::is_same<sid::strides_type<a_sid>, strides>{}, "");
        static_assert(std::is_same<sid::reference_type<a_sid, 3>, a_ref>{}, "");

        TEST(dummy, dummy) {}
    } // namespace
} // namespace gridtools
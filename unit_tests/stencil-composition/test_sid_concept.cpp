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

#include <gridtools/common/array.hpp>
#include <gridtools/stencil-composition/sid/concept.cpp>

#include <gtest/gtest.h>

namespace gridtools {
    namespace {
        // several primitive not sids
        static_assert(!is_sid<void>{}, "");
        static_assert(!is_sid<int>{}, "");
        struct garbage {};
        static_assert(!is_sid<garbage>{}, "");

        // test metafunctions on fully custom defined sid
        namespace custom {
            struct element {};
            struct ptr_diff {};
            struct ptr {
                element &operator*() const;
                friend ptr_diff operator-(ptr, ptr);
                friend ptr operator+(ptr, ptr_diff);
            };
            struct stride {
                friend void sid_shift(ptr &, stride const &, int);
            };
            struct bounds_validator {
                bool operator()(...) const;
            };

            struct strides_kind;
            struct bounds_validator_kind;

            struct testee {
                friend ptr sid_get_origin(testee);
                friend array<stride, 2> sid_get_strides(testee);
                friend bounds_validator sid_get_bounds_validator(testee);
                friend strides_kind sid_get_strides_kind(testee);
                friend bounds_validator_kind sid_get_bounds_validator_kind(testee);
            };

            static_assert(is_sid<testee>{}, "");
            static_assert(std::is_same<GT_META_CALL(sid::ptr_type, testee), ptr>{}, "");
            static_assert(std::is_same<GT_META_CALL(sid::ptr_diff_type, testee), ptr_diff>{}, "");
            static_assert(std::is_same<GT_META_CALL(sid::bounds_validator_type, testee), bounds_validator>{}, "");
            static_assert(std::is_same<GT_META_CALL(sid::reference_type, testee), element &>{}, "");
            static_assert(std::is_same<GT_META_CALL(sid::element_type, testee), element>{}, "");
            static_assert(std::is_same<GT_META_CALL(sid::const_reference_type, testee), element const &>{}, "");
            static_assert(std::is_same<GT_META_CALL(sid::strides_kind, testee), strides_kind>{}, "");
            static_assert(std::is_same<GT_META_CALL(sid::bounds_validator_kind, testee), bounds_validator_kind>{}, "");

            static_assert(
                std::is_same<decay_t<decltype(sid::get_stride<0>(sid::get_strides(testee{})))>, stride>{}, "");
            static_assert(
                std::is_same<decay_t<decltype(sid::get_stride<1>(sid::get_strides(testee{})))>, stride>{}, "");
            static_assert(
                !std::is_same<decay_t<decltype(sid::get_stride<2>(sid::get_strides(testee{})))>, stride>{}, "");
        } // namespace custom
    }     // namespace
} // namespace gridtools
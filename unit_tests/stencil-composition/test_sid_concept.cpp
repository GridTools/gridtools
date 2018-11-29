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

#include <gridtools/stencil-composition/sid/concept.hpp>

#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/array.hpp>
#include <gridtools/common/host_device.hpp>
#include <gridtools/common/tuple.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/meta/type_traits.hpp>

namespace gridtools {
    namespace {
        // several primitive not sids
        static_assert(!is_sid<void>(), "");
        static_assert(!is_sid<int>(), "");
        struct garbage {};
        static_assert(!is_sid<garbage>(), "");

        // fully custom defined sid
        namespace custom {
            struct element {};
            struct ptr_diff {
                int val;
            };
            struct ptr {
                element *val;
                GT_FUNCTION element &operator*() const { return *val; }
                friend GT_FUNCTION ptr operator+(ptr, ptr_diff) { return {}; }
            };
            struct stride {
                friend GT_FUNCTION std::true_type sid_shift(ptr &, stride const &, int) { return {}; }
                friend GT_FUNCTION std::false_type sid_shift(ptr_diff &, stride const &, int) { return {}; }
            };
            using strides = array<stride, 2>;
            struct bounds_validator {
                GT_FUNCTION std::false_type operator()(...) const { return {}; }
            };

            struct strides_kind;
            struct bounds_validator_kind;

            struct testee {
                friend GT_FUNCTION ptr sid_get_origin(testee &) { return {}; }
                friend GT_FUNCTION strides sid_get_strides(testee const &) { return {}; }
                friend GT_FUNCTION bounds_validator sid_get_bounds_validator(testee const &) { return {}; }

                friend ptr_diff sid_get_ptr_diff(testee);
                friend strides_kind sid_get_strides_kind(testee);
                friend bounds_validator_kind sid_get_bounds_validator_kind(testee);
            };

            static_assert(is_sid<testee>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::ptr_diff_type, testee), ptr_diff>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::bounds_validator_type, testee), bounds_validator>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::reference_type, testee), element &>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::element_type, testee), element>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::const_reference_type, testee), element const &>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::strides_kind, testee), strides_kind>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::bounds_validator_kind, testee), bounds_validator_kind>(), "");

            static_assert(std::is_same<decltype(sid::get_origin(std::declval<testee &>())), ptr>::value, "");
            static_assert(std::is_same<decltype(sid::get_strides(testee{})), strides>(), "");
            static_assert(std::is_same<decltype(sid::get_bounds_validator(testee{})), bounds_validator>(), "");

            static_assert(std::is_same<decay_t<decltype(sid::get_stride<0>(strides{}))>, stride>(), "");
            static_assert(std::is_same<decay_t<decltype(sid::get_stride<1>(strides{}))>, stride>(), "");
            static_assert(sid::get_stride<2>(strides{}) == 0, "");
            static_assert(sid::get_stride<42>(strides{}) == 0, "");

            static_assert(std::is_same<decltype(sid::shift(std::declval<ptr &>(), stride{}, 0)), std::true_type>(), "");
            static_assert(
                std::is_same<decltype(sid::shift(std::declval<ptr_diff &>(), stride{}, 0)), std::false_type>(), "");
        } // namespace custom

        namespace fallbacks {

            struct testee {
                friend GT_FUNCTION testee *sid_get_origin(testee &obj) { return &obj; }
            };

            static_assert(is_sid<testee>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::ptr_type, testee), testee *>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::ptr_diff_type, testee), ptrdiff_t>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::reference_type, testee), testee &>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::element_type, testee), testee>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::const_reference_type, testee), testee const &>(), "");

            using strides = GT_META_CALL(sid::strides_type, testee);
            static_assert(tuple_util::size<strides>() == 0, "");

            using bounds_validator = GT_META_CALL(sid::bounds_validator_type, testee);
            static_assert(bounds_validator{}(testee{}), "");

            static_assert(std::is_same<GT_META_CALL(sid::strides_kind, testee), strides>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::bounds_validator_kind, testee), bounds_validator>(), "");

            static_assert(std::is_same<decltype(sid::get_origin(std::declval<testee &>())), testee *>(), "");
            static_assert(std::is_same<decltype(sid::get_strides(testee{})), strides>(), "");
            static_assert(std::is_same<decltype(sid::get_bounds_validator(testee{})), bounds_validator>(), "");

            constexpr auto stride = sid::get_stride<0>(strides{});
            static_assert(stride == 0, "");
            static_assert(sid::get_stride<42>(strides{}) == 0, "");

            static_assert(std::is_void<void_t<decltype(sid::shift(std::declval<testee *&>(), stride, 42))>>(), "");
            static_assert(std::is_void<void_t<decltype(sid::shift(std::declval<ptrdiff_t *&>(), stride, 42))>>(), "");
        } // namespace fallbacks

        template <class T, class Stride, class Offset>
        void do_verify_shift(T obj, Stride stride, Offset offset) {
            auto expected = obj + stride * offset;
            sid::shift(obj, stride, offset);
            EXPECT_EQ(expected, obj);
        }

        struct verify_shift_f {
            template <class Stride, class Offset>
            void operator()(Stride stride, Offset offset) const {
                int const data[100] = {};
                do_verify_shift(data + 50, stride, offset);
                do_verify_shift(42, stride, offset);
            }
        };

        TEST(shift, default_overloads) {
            namespace tu = tuple_util;
            auto samples = tu::host_device::make<tuple>(
                2, 3, static_int<-2>{}, static_int<-1>{}, static_int<0>{}, static_int<1>{}, static_int<2>{});
            tu::host::for_each_in_cartesian_product(verify_shift_f{}, samples, samples);
        }
    } // namespace
} // namespace gridtools

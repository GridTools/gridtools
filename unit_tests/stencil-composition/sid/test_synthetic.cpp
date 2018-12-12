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

#include <gridtools/stencil-composition/sid/synthetic.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/array.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/meta/macros.hpp>
#include <gridtools/stencil-composition/sid/concept.hpp>

namespace gridtools {
    namespace {

        using sid::property;

        TEST(sid_synthetic, smoke) {
            double a = 100;
            auto testee = sid::synthetic().set<property::origin>(&a);
            static_assert(is_sid<decltype(testee)>::value, "");

            EXPECT_EQ(a, *sid::get_origin(testee));
        }

        namespace custom {
            struct element {};
            struct ptr_diff {
                int val;
            };
            struct ptr {
                element const *val;
                GT_FUNCTION element const &operator*() const { return *val; }
                friend GT_FUNCTION ptr operator+(ptr, ptr_diff) { return {}; }
            };
            struct stride {
                int val;
                friend GT_FUNCTION std::true_type sid_shift(ptr &, stride const &, int) { return {}; }
                friend GT_FUNCTION std::false_type sid_shift(ptr_diff &, stride const &, int) { return {}; }
            };
            using strides = array<stride, 2>;
            struct bounds_validator {
                int val;
                template <class T>
                GT_FUNCTION false_type operator()(T &&) const {
                    return {};
                }
            };

            struct strides_kind;
            struct bounds_validator_kind;

            TEST(sid_synthetic, custom) {
                element the_element = {};
                ptr the_origin = {&the_element};
                strides the_strides = {stride{3}, stride{4}};
                bounds_validator the_bounds_validator = {88};

                auto the_testee = sid::synthetic()
                                      .set<property::origin>(the_origin)
                                      .set<property::strides>(the_strides)
                                      .set<property::bounds_validator>(the_bounds_validator)
                                      .set<property::ptr_diff, ptr_diff>()
                                      .set<property::strides_kind, strides_kind>()
                                      .set<property::bounds_validator_kind, bounds_validator_kind>();

                using testee = decltype(the_testee);

                static_assert(is_sid<testee>(), "");
                static_assert(std::is_trivially_copyable<testee>(), "");

                static_assert(std::is_same<GT_META_CALL(sid::ptr_type, testee), ptr>(), "");
                static_assert(std::is_same<GT_META_CALL(sid::strides_type, testee), strides>(), "");
                static_assert(std::is_same<GT_META_CALL(sid::bounds_validator_type, testee), bounds_validator>(), "");
                static_assert(std::is_same<GT_META_CALL(sid::ptr_diff_type, testee), ptr_diff>(), "");
                static_assert(std::is_same<GT_META_CALL(sid::strides_kind, testee), strides_kind>(), "");
                static_assert(
                    std::is_same<GT_META_CALL(sid::bounds_validator_kind, testee), bounds_validator_kind>(), "");

                EXPECT_EQ(&the_element, sid::get_origin(the_testee).val);
                EXPECT_EQ(3, tuple_util::get<0>(sid::get_strides(the_testee)).val);
                EXPECT_EQ(4, tuple_util::get<1>(sid::get_strides(the_testee)).val);
                EXPECT_EQ(88, sid::get_bounds_validator(the_testee).val);
            }
        } // namespace custom
    }     // namespace
} // namespace gridtools

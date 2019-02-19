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

#include <gridtools/stencil-composition/sid/loop.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/compose.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/stencil-composition/sid/synthetic.hpp>

namespace gridtools {
    namespace {
        using sid::property;
        using namespace literals;
        namespace tu = tuple_util::host_device;

        struct assignment_f {
            double m_val;
            template <class Strides>
            GT_FUNCTION void operator()(double *ptr, Strides const &) const {
                *ptr = m_val;
            }
        };

        using i_t = integral_constant<int, 0>;
        using j_t = integral_constant<int, 1>;

        TEST(make_loop, smoke) {
            double data[10][10] = {};
            auto strides = tu::make<tuple>(10_c, 1_c);
            using strides_t = decltype(strides);

            double *ptr = &data[0][0];
            sid::make_loop<i_t>(5, 1)(assignment_f{42})(ptr, strides);
            for (int i = 0; i < 5; ++i)
                EXPECT_EQ(42, data[i][0]) << " i:" << i;

            ptr = &data[2][3];
            sid::make_loop<j_t>(4_c, -1_c)(assignment_f{5})(ptr, strides);
            for (int i = 0; i < 4; ++i)
                EXPECT_EQ(5, data[2][i]) << " i:" << i;

            ptr = &data[0][0];
            sid::make_loop<i_t>(10_c)(sid::make_loop<j_t>(10_c)(assignment_f{88}))(ptr, strides);
            for (int i = 0; i < 10; ++i)
                for (int j = 0; j < 10; ++j)
                    EXPECT_EQ(88, data[i][j]) << " i:" << i << ", j:" << j;
        }

        TEST(nest_loops, smoke) {
            double data[10][10] = {};
            double *ptr = &data[0][0];
            auto strides = tu::make<tuple>(10_c, 1_c);
            using strides_t = decltype(strides);

            auto testee = host_device::compose(sid::make_loop<i_t>(10_c), sid::make_loop<j_t>(10_c));

            testee(assignment_f{42})(ptr, strides);

            for (int i = 0; i < 10; ++i)
                for (int j = 0; j < 10; ++j)
                    EXPECT_EQ(42, data[i][j]) << " i:" << i << ", j:" << j;
        }

        TEST(range, smoke) {
            double data[10][10] = {};
            double *ptr = &data[0][0];
            auto strides = tu::make<tuple>(10_c, 1_c);

            auto testee = sid::make_range(ptr, strides, sid::make_loop<i_t>(10_c), sid::make_loop<j_t>(10_c));

            for (auto &&val : testee)
                val = 42;

            for (int i = 0; i < 10; ++i)
                for (int j = 0; j < 10; ++j)
                    EXPECT_EQ(42, data[i][j]) << " i:" << i << ", j:" << j;
        }
    } // namespace
} // namespace gridtools

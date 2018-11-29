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

#include <gridtools/common/generic_metafunctions/for_each.hpp>

#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <gridtools/common/host_device.hpp>

namespace gridtools {

    struct f {
        int *&dst;

        template <class T>
        GT_FUNCTION_WARNING void operator()(T) const {
            *(dst++) = T::value;
        }
    };

    struct ff {
        int *&dst;

        template <class T>
        GT_FUNCTION_WARNING void operator()() const {
            *(dst++) = T::value;
        }
    };

    template <class...>
    struct lst;

    template <int I>
    using my_int_t = std::integral_constant<int, I>;

    TEST(for_each, empty) {
        int vals[3];
        int *cur = vals;
        for_each<lst<>>(f{cur});
        EXPECT_EQ(cur, cur);
    }

    TEST(for_each, functional) {
        int vals[3];
        int *cur = vals;
        for_each<lst<my_int_t<0>, my_int_t<42>, my_int_t<3>>>(f{cur});
        EXPECT_EQ(cur, vals + 3);
        EXPECT_THAT(vals, testing::ElementsAre(0, 42, 3));
    }

    TEST(for_each_type, functional) {
        int vals[3];
        int *cur = vals;
        for_each_type<lst<my_int_t<0>, my_int_t<42>, my_int_t<3>>>(ff{cur});
        EXPECT_EQ(cur, vals + 3);
        EXPECT_THAT(vals, testing::ElementsAre(0, 42, 3));
    }

    TEST(for_each, targets) {
        int *ptr = nullptr;
        for_each<lst<>>(f{ptr});
        host::for_each<lst<>>(f{ptr});
        host_device::for_each<lst<>>(f{ptr});
    }
} // namespace gridtools

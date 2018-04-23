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

#include <array>
#include <tuple>
#include <utility>

#include <common/tuple_util.hpp>

#include <gtest/gtest.h>

#include <common/defs.hpp>

#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ < 9
#define NO_CONSTEXPR
#endif

#ifdef NO_CONSTEXPR
#define CONSTEXPR
#else
#define CONSTEXPR constexpr
#endif

namespace custom {
    struct foo {
        int a;
        double b;

        friend CONSTEXPR int do_get(std::integral_constant< size_t, 0 >, foo const &obj) { return obj.a; }
        friend int &do_get(std::integral_constant< size_t, 0 >, foo &obj) { return obj.a; }
        friend CONSTEXPR int do_get(std::integral_constant< size_t, 0 >, foo &&obj) { return obj.a; }
        friend CONSTEXPR double do_get(std::integral_constant< size_t, 1 >, foo const &obj) { return obj.b; }
        friend double &do_get(std::integral_constant< size_t, 1 >, foo &obj) { return obj.b; }
        friend CONSTEXPR double do_get(std::integral_constant< size_t, 1 >, foo &&obj) { return obj.b; }
    };
}

namespace gridtools {
    namespace tuple_util {

        TEST(get, custom) {
            custom::foo obj{1, 2};
            EXPECT_EQ(get< 0 >(obj), 1);
            EXPECT_EQ(get< 1 >(obj), 2);
            get< 0 >(obj) = 42;
            EXPECT_EQ(get< 0 >(obj), 42);

#ifndef NO_CONSTEXPR
            constexpr custom::foo c_obj{2, 4};
            static_assert(get< 0 >(c_obj) == 2, "");
            static_assert(get< 0 >(custom::foo{3}) == 3, "");
#endif
        }

        TEST(get, std_tuple) {
            auto obj = std::make_tuple(1, 2.);
            EXPECT_EQ(get< 0 >(obj), 1);
            EXPECT_EQ(get< 1 >(obj), 2);
            get< 0 >(obj) = 42;
            EXPECT_EQ(get< 0 >(obj), 42);
        }

        TEST(get, std_pair) {
            auto obj = std::make_pair(1, 2.);
            EXPECT_EQ(get< 0 >(obj), 1);
            EXPECT_EQ(get< 1 >(obj), 2);
            get< 0 >(obj) = 42;
            EXPECT_EQ(get< 0 >(obj), 42);
        }

        TEST(get, std_array) {
            auto obj = std::array< int, 2 >{1, 2};
            EXPECT_EQ(get< 0 >(obj), 1);
            EXPECT_EQ(get< 1 >(obj), 2);
            get< 0 >(obj) = 42;
            EXPECT_EQ(get< 0 >(obj), 42);
        }

        struct add_2_f {
            template < class T >
            T operator()(T val) const {
                return val + 2;
            }
        };

        TEST(transform, functional) {
            auto src = std::make_tuple(42, 5.3);
            auto res = transform(add_2_f{}, src);
            static_assert(std::is_same< decltype(res), decltype(src) >{}, "");
            EXPECT_EQ(res, std::make_tuple(44, 7.3));
        }

        struct accumulate_f {
            double &m_acc;
            template < class T >
            void operator()(T val) const {
                m_acc += val;
            }
        };

        TEST(for_each, functional) {
            auto src = std::make_tuple(42, 5.3);
            double acc = 0;
            for_each(accumulate_f{acc}, src);
            EXPECT_EQ(47.3, acc);
        }

        TEST(flatten, functional) {
            EXPECT_EQ(
                flatten(std::make_tuple(std::make_tuple(1, 2), std::make_tuple(3, 4))), std::make_tuple(1, 2, 3, 4));
        }

        TEST(flatten, ref) {
            auto orig = std::make_tuple(std::make_tuple(1, 2), std::make_tuple(3, 4));
            auto flat = flatten(orig);
            EXPECT_EQ(flat, std::make_tuple(1, 2, 3, 4));
            get< 0 >(flat) = 42;
            EXPECT_EQ(get< 0 >(get< 0 >(orig)), 42);
        }

        TEST(drop_front, functional) { EXPECT_EQ(drop_front< 2 >(std::make_tuple(1, 2, 3, 4)), std::make_tuple(3, 4)); }

        TEST(push_back, functional) { EXPECT_EQ(push_back(std::make_tuple(1, 2), 3, 4), std::make_tuple(1, 2, 3, 4)); }

        TEST(fold, functional) {
            auto f = [](int x, int y) { return x + y; };
            EXPECT_EQ(fold(f, std::make_tuple(1, 2, 3, 4)), 10);
        }
    }
}

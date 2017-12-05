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

#include <c_bindings/function_wrapper.hpp>

#include <type_traits>
#include <stack>

#include <gtest/gtest.h>

namespace gridtools {
    namespace c_bindings {
        namespace {

            struct a_struct;
            static_assert(std::is_same< wrapped_t< void() >, void() >{}, "");
            static_assert(std::is_same< wrapped_t< int() >, int() >{}, "");
            static_assert(std::is_same< wrapped_t< const int() >, const int() >{}, "");
            static_assert(std::is_same< wrapped_t< a_struct() >, gt_handle *() >{}, "");
            static_assert(std::is_same< wrapped_t< a_struct const &() >, gt_handle *() >{}, "");
            static_assert(std::is_same< wrapped_t< void(int) >, void(int) >{}, "");
            static_assert(std::is_same< wrapped_t< void(int &) >, void(int *) >{}, "");
            static_assert(std::is_same< wrapped_t< void(int *) >, void(int *) >{}, "");
            static_assert(std::is_same< wrapped_t< void(a_struct *) >, void(gt_handle *) >{}, "");
            static_assert(std::is_same< wrapped_t< void(a_struct &) >, void(gt_handle *) >{}, "");
            static_assert(std::is_same< wrapped_t< void(a_struct) >, void(gt_handle *) >{}, "");

            template < class T >
            std::stack< T > create() {
                return {};
            }

            template < class T >
            void push(std::stack< T > &obj, T val) {
                obj.push(val);
            }

            template < class T >
            void pop(std::stack< T > &obj) {
                obj.pop();
            }

            template < class T >
            T top(const std::stack< T > &obj) {
                return obj.top();
            }

            template < class T >
            bool empty(const std::stack< T > &obj) {
                return obj.empty();
            }

            TEST(wrap, smoke) {
                gt_handle *obj = wrap(create< int >)();
                EXPECT_TRUE(wrap(empty< int >)(obj));
                wrap(push< int >)(obj, 42);
                EXPECT_FALSE(wrap(empty< int >)(obj));
                EXPECT_EQ(42, wrap(top< int >)(obj));
                wrap(pop< int >)(obj);
                EXPECT_TRUE(wrap(empty< int >)(obj));
                gt_release(obj);
            }

            void inc(int &val) { ++val; }

            TEST(wrap, const_expr) {
                constexpr auto wrapped_inc = wrap(inc);
                int i = 41;
                wrapped_inc(&i);
                EXPECT_EQ(42, i);
            }

            TEST(warp, lambda) {
                EXPECT_EQ(42, wrap(+[] { return 42; })());
            }
        }
    }
}

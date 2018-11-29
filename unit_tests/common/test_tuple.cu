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

#include <gridtools/common/tuple.hpp>

#include <gtest/gtest.h>

#include "../../include/gridtools/common/tuple_util.hpp"
#include <gridtools/common/cuda_util.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/meta/type_traits.hpp>

namespace gridtools {
    namespace on_device {
        namespace {

            template <class Res, class Fun, class... Args>
            __global__ void kernel(Res *res, Fun fun, Args... args) {
                *res = fun(args...);
            }

            template <class Fun, class... Args, class Res = decay_t<result_of_t<Fun(Args...)>>>
            Res exec(Fun fun, Args... args) {
                auto res = cuda_util::make_clone(Res{});
                kernel<<<1, 1>>>(res.get(), fun, args...);
                return cuda_util::from_clone(res);
            }

            TEST(tuple, get) {
                tuple<int, double> src{42, 2.5};
                EXPECT_EQ(42, exec(tuple_util::device::get_nth_f<0>{}, src));
                EXPECT_EQ(2.5, exec(tuple_util::device::get_nth_f<1>{}, src));
            }

            __device__ tuple<int, double> element_wise_ctor(int x, double y) { return {x, y}; }

            TEST(tuple, element_wise_ctor) {
                tuple<int, double> testee = exec(element_wise_ctor, 42, 2.5);
                EXPECT_EQ(42, tuple_util::host::get<0>(testee));
                EXPECT_EQ(2.5, tuple_util::host::get<1>(testee));
            }

            __device__ tuple<int, double> element_wise_conversion_ctor(char x, char y) { return {x, y}; }

            TEST(tuple, element_wise_conversion_ctor) {
                tuple<int, double> testee = exec(element_wise_conversion_ctor, 'a', 'b');
                EXPECT_EQ('a', tuple_util::host::get<0>(testee));
                EXPECT_EQ('b', tuple_util::host::get<1>(testee));
            }

            __device__ tuple<int, double> tuple_conversion_ctor(tuple<char, char> const &src) { return src; }

            TEST(tuple, tuple_conversion_ctor) {
                tuple<int, double> testee = exec(tuple_conversion_ctor, tuple<char, char>{'a', 'b'});
                EXPECT_EQ('a', tuple_util::host::get<0>(testee));
                EXPECT_EQ('b', tuple_util::host::get<1>(testee));
            }

        } // namespace
    }     // namespace on_device
} // namespace gridtools

#include "test_tuple.cpp"

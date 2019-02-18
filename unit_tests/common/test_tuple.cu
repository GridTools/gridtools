/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/tuple.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/cuda_util.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/meta/type_traits.hpp>

namespace gridtools {
    namespace on_device {
        template <class Res, class Fun, class... Args>
        __global__ void kernel(Res *res, Fun fun, Args... args) {
            *res = fun(args...);
        }

        template <class Fun, class... Args, class Res = decay_t<result_of_t<Fun(Args...)>>>
        Res exec(Fun fun, Args... args) {
            static_assert(!std::is_pointer<Fun>::value, "");
            static_assert(conjunction<negation<std::is_pointer<Args>>...>::value, "");
            static_assert(std::is_trivially_copyable<Res>::value, "");
            auto res = cuda_util::cuda_malloc<Res>();
            kernel<<<1, 1>>>(res.get(), fun, args...);
            GT_CUDA_CHECK(cudaDeviceSynchronize());
            return cuda_util::from_clone(res);
        }

        TEST(tuple, get) {
            tuple<int, double> src{42, 2.5};
            EXPECT_EQ(42, exec(tuple_util::device::get_nth_f<0>{}, src));
            EXPECT_EQ(2.5, exec(tuple_util::device::get_nth_f<1>{}, src));
        }

        __device__ tuple<int, double> element_wise_ctor(int x, double y) { return {x, y}; }

#define MAKE_CONSTANT(fun) integral_constant<decltype(&fun), &fun>()

        TEST(tuple, element_wise_ctor) {
            tuple<int, double> testee = exec(MAKE_CONSTANT(element_wise_ctor), 42, 2.5);
            EXPECT_EQ(42, tuple_util::host::get<0>(testee));
            EXPECT_EQ(2.5, tuple_util::host::get<1>(testee));
        }

        __device__ tuple<int, double> element_wise_conversion_ctor(char x, char y) { return {x, y}; }

        TEST(tuple, element_wise_conversion_ctor) {
            tuple<int, double> testee = exec(MAKE_CONSTANT(element_wise_conversion_ctor), 'a', 'b');
            EXPECT_EQ('a', tuple_util::host::get<0>(testee));
            EXPECT_EQ('b', tuple_util::host::get<1>(testee));
        }

        __device__ tuple<int, double> tuple_conversion_ctor(tuple<char, char> const &src) { return src; }

        TEST(tuple, tuple_conversion_ctor) {
            tuple<int, double> testee = exec(MAKE_CONSTANT(tuple_conversion_ctor), tuple<char, char>{'a', 'b'});
            EXPECT_EQ('a', tuple_util::host::get<0>(testee));
            EXPECT_EQ('b', tuple_util::host::get<1>(testee));
        }

#undef MAKE_CONSTANT
    } // namespace on_device
} // namespace gridtools

#include "test_tuple.cpp"

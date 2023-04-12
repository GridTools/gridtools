/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/dimension_to_tuple_like.hpp>
#include <gridtools/stencil/cartesian.hpp>

#include <stencil_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;

    struct copy_functor_unrolled {
        using in0 = in_accessor<0>;
        using in1 = in_accessor<1>;
        using in2 = in_accessor<2>;
        using in3 = in_accessor<3>;
        using in4 = in_accessor<4>;
        using out0 = inout_accessor<5>;
        using out1 = inout_accessor<6>;
        using out2 = inout_accessor<7>;
        using out3 = inout_accessor<8>;
        using out4 = inout_accessor<9>;

        using param_list = make_param_list<in0, in1, in2, in3, in4, out0, out1, out2, out3, out4>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            eval(out0()) = eval(in0());
            eval(out1()) = eval(in1());
            eval(out2()) = eval(in2());
            eval(out3()) = eval(in3());
            eval(out4()) = eval(in4());
        }
    };

    struct copy_functor_tuple {
        using in = in_accessor<0>;
        using out = inout_accessor<1>;

        using param_list = make_param_list<in, out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            eval(out()) = eval(in());
        }
    };

    struct copy_functor_tuple_unrolled {
        using in = in_accessor<0>;
        using out = inout_accessor<1>;

        using param_list = make_param_list<in, out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            tuple_util::host_device::get<0>(eval(out())) = tuple_util::host_device::get<0>(eval(in()));
            tuple_util::host_device::get<1>(eval(out())) = tuple_util::host_device::get<1>(eval(in()));
            tuple_util::host_device::get<2>(eval(out())) = tuple_util::host_device::get<2>(eval(in()));
            tuple_util::host_device::get<3>(eval(out())) = tuple_util::host_device::get<3>(eval(in()));
            tuple_util::host_device::get<4>(eval(out())) = tuple_util::host_device::get<4>(eval(in()));
        }
    };

    struct copy_functor_data_dims {
        using in = in_accessor<0, extent<>, 4>;
        using out = inout_accessor<1, extent<>, 4>;

        using param_list = make_param_list<in, out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            eval(out(0, 0, 0, 0)) = eval(in(0, 0, 0, 0));
            eval(out(0, 0, 0, 1)) = eval(in(0, 0, 0, 1));
            eval(out(0, 0, 0, 2)) = eval(in(0, 0, 0, 2));
            eval(out(0, 0, 0, 3)) = eval(in(0, 0, 0, 3));
            eval(out(0, 0, 0, 4)) = eval(in(0, 0, 0, 4));
        }
    };

    GT_REGRESSION_TEST(copy_stencil_tuple_unrolled, test_environment<>, stencil_backend_t) {
        // baseline
        auto in = [](int i, int j, int k) { return i + j + k; };
        auto out0 = TypeParam::make_storage();
        auto out1 = TypeParam::make_storage();
        auto out2 = TypeParam::make_storage();
        auto out3 = TypeParam::make_storage();
        auto out4 = TypeParam::make_storage();
        auto comp = [&out0,
                        &out1,
                        &out2,
                        &out3,
                        &out4,
                        grid = TypeParam::make_grid(),
                        in0 = TypeParam::make_const_storage(in),
                        in1 = TypeParam::make_const_storage(in),
                        in2 = TypeParam::make_const_storage(in),
                        in3 = TypeParam::make_const_storage(in),
                        in4 = TypeParam::make_const_storage(in)] {
            run_single_stage(copy_functor_unrolled(),
                stencil_backend_t(),
                grid,
                in0,
                in1,
                in2,
                in3,
                in4,
                out0,
                out1,
                out2,
                out3,
                out4);
        };
        comp();
        TypeParam::verify(in, out0);
        TypeParam::verify(in, out1);
        TypeParam::verify(in, out2);
        TypeParam::verify(in, out3);
        TypeParam::verify(in, out4);
        TypeParam::benchmark("copy_stencil_tuple_unrolled", comp);
    }
    GT_REGRESSION_TEST(copy_stencil_tuple_array, test_environment<>, stencil_backend_t) {
        // expected to perform bad on GPU because array of structures (if element > 128 bits)
        using float_t = typename TypeParam::float_t;
        auto in = [](int i, int j, int k) {
            return array<float_t, 5>{//
                float_t(i + j + k),
                float_t(i + j + k + 1),
                float_t(i + j + k + 2),
                float_t(i + j + k + 3),
                float_t(i + j + k + 4)};
        };
        auto out = TypeParam::template make_storage<array<float_t, 5>>();
        auto comp =
            [&out, grid = TypeParam::make_grid(), in = TypeParam::template make_const_storage<array<float_t, 5>>(in)] {
                run_single_stage(copy_functor_tuple(), stencil_backend_t(), grid, in, out);
            };
        comp();
        TypeParam::verify(in, out);
        TypeParam::benchmark("copy_stencil_tuple_array", comp);
    }

    GT_REGRESSION_TEST(copy_stencil_tuple_extra_dim, test_environment<>, stencil_backend_t) {
        // extra dimension without tuple behavior
        using float_t = typename TypeParam::float_t;
        auto in = [](int i, int j, int k, int t) { return i + j + k + t; };
        auto out = TypeParam::template builder<float_t>(integral_constant<int, 5>{}).build();
        auto comp =
            [&out,
                grid = TypeParam::make_grid(),
                in = TypeParam::template builder<float_t const>(integral_constant<int, 5>{}).initializer(in).build()] {
                run_single_stage(copy_functor_data_dims(), stencil_backend_t(), grid, in, out);
            };
        comp();
        TypeParam::verify(in, out);
        TypeParam::benchmark("copy_stencil_tuple_extra_dim", comp);
    }

    GT_REGRESSION_TEST(copy_stencil_tuple_composite, test_environment<>, stencil_backend_t) {
        auto in = [](int i, int j, int k) { return i + j + k; };
        auto out0 = TypeParam::make_storage();
        auto out1 = TypeParam::make_storage();
        auto out2 = TypeParam::make_storage();
        auto out3 = TypeParam::make_storage();
        auto out4 = TypeParam::make_storage();
        auto out = sid::composite::keys< //
            integral_constant<int, 0>,
            integral_constant<int, 1>,
            integral_constant<int, 2>,
            integral_constant<int, 3>,
            integral_constant<int, 4>>::make_values(out0, out1, out2, out3, out4);
        auto in0 = TypeParam::make_const_storage(in);
        auto in1 = TypeParam::make_const_storage(in);
        auto in2 = TypeParam::make_const_storage(in);
        auto in3 = TypeParam::make_const_storage(in);
        auto in4 = TypeParam::make_const_storage(in);
        auto in_sid = sid::composite::keys< //
            integral_constant<int, 0>,
            integral_constant<int, 1>,
            integral_constant<int, 2>,
            integral_constant<int, 3>,
            integral_constant<int, 4>>::make_values(in0, in1, in2, in3, in4);
        auto comp = [&out, grid = TypeParam::make_grid(), &in = in_sid] {
            run_single_stage(copy_functor_tuple(), stencil_backend_t(), grid, in, out);
        };
        comp();
        TypeParam::verify(in, out0);
        TypeParam::verify(in, out1);
        TypeParam::verify(in, out2);
        TypeParam::verify(in, out3);
        TypeParam::verify(in, out4);
        TypeParam::benchmark("copy_stencil_tuple_composite", comp);
    }

    GT_REGRESSION_TEST(copy_stencil_tuple_dim2tuple, test_environment<>, stencil_backend_t) {
        using float_t = typename TypeParam::float_t;
        auto in = [](int i, int j, int k, int t) { return i + j + k + t; };
        auto out_ds = TypeParam::template builder<float_t>(integral_constant<int, 5>{}).build();
        auto out = sid::dimension_to_tuple_like<integral_constant<int, 3>, 5>(out_ds);
        auto in_ds = TypeParam::template builder<float_t const>(integral_constant<int, 5>{}).initializer(in).build();
        auto in_transformed = sid::dimension_to_tuple_like<integral_constant<int, 3>, 5>(in_ds);
        auto comp = [&out, grid = TypeParam::make_grid(), &in = in_transformed] {
            run_single_stage(copy_functor_tuple(), stencil_backend_t(), grid, in, out);
        };
        comp();
        TypeParam::verify(in, out_ds);
        TypeParam::benchmark("copy_stencil_tuple_dim2tuple", comp);
    }

} // namespace

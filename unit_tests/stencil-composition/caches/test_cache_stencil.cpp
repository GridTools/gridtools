/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <functional>

#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/computation_fixture.hpp>

namespace gridtools {
    namespace {

        struct cache_stencil : computation_fixture<1> {
            cache_stencil() : computation_fixture<1>{128, 128, 30} {}

            ~cache_stencil() { verify(make_storage(expected), out); }

            using fun_t = std::function<float_type(int, int, int)>;

            fun_t in = [](int i, int j, int k) { return i + j * 100 + k * 10000; };

            storage_type out = make_storage();

            fun_t expected;
        };

        struct functor1 {
            using in = in_accessor<0>;
            using out = inout_accessor<1>;
            using param_list = make_param_list<in, out>;

            template <typename Evaluation>
            GT_FUNCTION static void apply(Evaluation &eval) {
                eval(out()) = eval(in());
            }
        };

        TEST_F(cache_stencil, ij_cache) {
            make_computation(p_0 = make_storage(in),
                p_1 = out,
                make_multistage(execute::parallel(),
                    define_caches(cache<cache_type::ij, cache_io_policy::local>(p_tmp_0)),
                    make_stage<functor1>(p_0, p_tmp_0),
                    make_stage<functor1>(p_tmp_0, p_1)))
                .run();

            expected = in;
        }

        struct functor2 {
            using in = in_accessor<0, extent<-1, 1, -1, 1>>;
            using out = inout_accessor<1>;
            using param_list = make_param_list<in, out>;

            template <typename Evaluation>
            GT_FUNCTION static void apply(Evaluation &eval) {
                eval(out()) =
                    (eval(in(-1, 0, 0)) + eval(in(1, 0, 0)) + eval(in(0, -1, 0)) + eval(in(0, 1, 0))) / (float_type)4.0;
            }
        };

        TEST_F(cache_stencil, ij_cache_offset) {
            make_computation(p_0 = make_storage(in),
                p_1 = out,
                make_multistage(execute::parallel(),
                    define_caches(cache<cache_type::ij, cache_io_policy::local>(p_tmp_0)),
                    make_stage<functor1>(p_0, p_tmp_0),
                    make_stage<functor2>(p_tmp_0, p_1)))
                .run();

            expected = [this](int i, int j, int k) {
                return (in(i - 1, j, k) + in(i + 1, j, k) + in(i, j - 1, k) + in(i, j + 1, k)) / (float_type)4.0;
            };
        }

        struct functor3 {
            using in = in_accessor<0>;
            using out = inout_accessor<1>;
            using param_list = make_param_list<in, out>;

            template <typename Evaluation>
            GT_FUNCTION static void apply(Evaluation &eval) {
                eval(out()) = eval(in()) + 1;
            }
        };

        TEST_F(cache_stencil, multi_cache) {
            make_computation(p_0 = make_storage(in),
                p_1 = out,
                make_multistage(execute::parallel(),
                    define_caches(cache<cache_type::ij, cache_io_policy::local>(p_tmp_0, p_tmp_1),
                        cache<cache_type::ij, cache_io_policy::local>(p_tmp_2)),
                    make_stage<functor3>(p_0, p_tmp_0),
                    make_stage<functor3>(p_tmp_0, p_tmp_1),
                    make_stage<functor3>(p_tmp_1, p_tmp_2),
                    make_stage<functor3>(p_tmp_2, p_1)))
                .run();

            expected = [this](int i, int j, int k) { return in(i, j, k) + 4; };
        }
    } // namespace
} // namespace gridtools

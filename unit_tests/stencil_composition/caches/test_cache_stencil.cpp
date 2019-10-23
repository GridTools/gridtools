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

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/computation_fixture.hpp>

namespace gridtools {
    namespace {

        struct cache_stencil : computation_fixture<1> {
            cache_stencil() : computation_fixture<1>(128, 128, 30) {}

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
            run(
                [](auto in, auto out) {
                    GT_DECLARE_TMP(float_type, tmp);
                    return execute_parallel().ij_cached(tmp).stage(functor1(), in, tmp).stage(functor1(), tmp, out);
                },
                backend_t(),
                make_grid(),
                make_storage(in),
                out);
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
            run(
                [](auto in, auto out) {
                    GT_DECLARE_TMP(float_type, tmp);
                    return execute_parallel().ij_cached(tmp).stage(functor1(), in, tmp).stage(functor2(), tmp, out);
                },
                backend_t(),
                make_grid(),
                make_storage(in),
                out);

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
            run(
                [](auto in, auto out) {
                    GT_DECLARE_TMP(float_type, tmp0, tmp1, tmp2);
                    return execute_parallel()
                        .ij_cached(tmp0, tmp1)
                        .stage(functor3(), in, tmp0)
                        .stage(functor3(), tmp0, tmp1)
                        .stage(functor3(), tmp1, tmp2)
                        .stage(functor3(), tmp2, out);
                },
                backend_t(),
                make_grid(),
                make_storage(in),
                out);

            expected = [this](int i, int j, int k) { return in(i, j, k) + 4; };
        }
    } // namespace
} // namespace gridtools

/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/fn/unstructured2.hpp>

#include <gtest/gtest.h>

#include <gridtools/fn/backend2/naive.hpp>
#include <gridtools/sid/synthetic.hpp>
#include <gridtools/stencil/positional.hpp>

namespace gridtools::fn {
    namespace {
        using namespace literals;
        using sid::property;

        template <class V2V>
        struct stencil {
            constexpr auto operator()() const {
                return [](auto const &in) {
                    int sum = deref(in);
                    if (can_deref(shift(in, V2V(), 0)))
                        sum += deref(shift(in, V2V(), 0));
                    if (can_deref(shift(in, V2V(), 1)))
                        sum += deref(shift(in, V2V(), 1));
                    if (can_deref(shift(in, V2V(), 2)))
                        sum += deref(shift(in, V2V(), 2));
                    return sum;
                };
            }
        };

        struct vertex {};
        struct v2v {};

        TEST(unstructured, stencil) {
            auto apply_stencil = [](auto executor, auto &out, auto const &in, auto const &vidx) {
                executor().arg(out).arg(in).arg(vidx).assign(0_c, stencil<v2v>(), 1_c);
            };
            auto fencil =
                [&](auto const &v2v_table, int nvertices, int nlevels, auto &out, auto const &in, auto const &vidx) {
                    auto v2v_conn = connectivity<v2v, vertex, vertex>(v2v_table, 3_c);
                    auto domain = unstructured_domain<vertex>(nvertices, nlevels, v2v_conn);
                    auto backend = make_backend(backend::naive(), domain);
                    apply_stencil(backend.stencil_executor(), out, in, vidx);
                };

            int v2v_table[3][3] = {{1, 2, -1}, {0, 2, -1}, {0, 1, -1}};
            auto v2v_conn =
                sid::synthetic()
                    .set<property::origin>(sid::host::make_simple_ptr_holder(&v2v_table[0][0]))
                    .set<property::strides>(hymap::keys<vertex, unstructured::dim::neighbor>::values(3_c, 1_c));

            int in[3][5], out[3][5] = {};
            for (int v = 0; v < 3; ++v)
                for (int k = 0; k < 5; ++k)
                    in[v][k] = 5 * v + k;

            auto as_synthetic = [](int x[3][5]) {
                return sid::synthetic()
                    .set<property::origin>(sid::host::make_simple_ptr_holder(&x[0][0]))
                    .set<property::strides>(hymap::keys<vertex, unstructured::dim::k>::values(5_c, 1_c));
            };
            auto in_s = as_synthetic(in);
            auto out_s = as_synthetic(out);

            auto vidx = gridtools::stencil::positional<vertex>();

            fencil(v2v_conn, 3, 5, out_s, in_s, vidx);

            for (int v = 0; v < 3; ++v)
                for (int k = 0; k < 5; ++k) {
                    int nbsum = in[v][k];
                    for (int i = 0; i < 3; ++i) {
                        int nb = v2v_table[v][i];
                        if (nb != -1)
                            nbsum += in[nb][k];
                    }
                    EXPECT_EQ(out[v][k], nbsum);
                }
        }

    } // namespace
} // namespace gridtools::fn

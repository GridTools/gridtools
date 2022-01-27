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

namespace gridtools::fn {
    namespace {
        using namespace literals;
        using sid::property;

        template <class C>
        struct stencil {
            constexpr auto operator()() const {
                return [](auto const &in) { return reduce(C(), std::plus(), 0, in); };
            }
        };

        struct vertex {};
        struct edge {};
        struct v2v {
            static constexpr int max_neighbors = 3;
        };
        struct v2e {
            static constexpr int max_neighbors = 2;
        };

        TEST(unstructured, v2v_sum) {
            auto apply_stencil = [](auto executor, auto &out, auto const &in) {
                executor().arg(out).arg(in).assign(0_c, stencil<v2v>(), 1_c);
            };
            auto fencil = [&](auto const &v2v_table, int nvertices, int nlevels, auto &out, auto const &in) {
                auto v2v_conn = connectivity<v2v, vertex, vertex>(v2v_table);
                auto domain = unstructured_domain<vertex>(nvertices, nlevels, v2v_conn);
                auto backend = make_backend(backend::naive(), domain);
                apply_stencil(backend.stencil_executor(), out, in);
            };

            int v2v_table[3][3] = {{1, 2, -1}, {0, 2, -1}, {0, 1, -1}};
            auto v2v_conn =
                sid::synthetic()
                    .set<property::origin>(sid::host::make_simple_ptr_holder(&v2v_table[0][0]))
                    .set<property::strides>(hymap::keys<vertex, unstructured::dim::neighbor>::make_values(3_c, 1_c));

            int in[3][5], out[3][5] = {};
            for (int v = 0; v < 3; ++v)
                for (int k = 0; k < 5; ++k)
                    in[v][k] = 5 * v + k;

            auto as_synthetic = [](int x[3][5]) {
                return sid::synthetic()
                    .set<property::origin>(sid::host::make_simple_ptr_holder(&x[0][0]))
                    .set<property::strides>(hymap::keys<vertex, unstructured::dim::k>::make_values(5_c, 1_c));
            };
            auto in_s = as_synthetic(in);
            auto out_s = as_synthetic(out);

            fencil(v2v_conn, 3, 5, out_s, in_s);

            for (int v = 0; v < 3; ++v)
                for (int k = 0; k < 5; ++k) {
                    int nbsum = 0;
                    for (int i = 0; i < 3; ++i) {
                        int nb = v2v_table[v][i];
                        if (nb != -1)
                            nbsum += in[nb][k];
                    }
                    EXPECT_EQ(out[v][k], nbsum);
                }
        }

        TEST(unstructured, v2e_sum) {
            auto apply_stencil = [](auto executor, auto &out, auto const &in) {
                executor().arg(out).arg(in).assign(0_c, stencil<v2e>(), 1_c);
            };
            auto fencil = [&](auto const &v2e_table, int nvertices, int nlevels, auto &out, auto const &in) {
                auto v2e_conn = connectivity<v2e, vertex, edge>(v2e_table);
                auto domain = unstructured_domain<vertex>(nvertices, nlevels, v2e_conn);
                auto backend = make_backend(backend::naive(), domain);
                apply_stencil(backend.stencil_executor(), out, in);
            };

            int v2e_table[3][2] = {{0, 2}, {0, 1}, {1, 2}};
            auto v2e_conn =
                sid::synthetic()
                    .set<property::origin>(sid::host::make_simple_ptr_holder(&v2e_table[0][0]))
                    .set<property::strides>(hymap::keys<vertex, unstructured::dim::neighbor>::make_values(2_c, 1_c));

            int in[3][5], out[3][5] = {};
            for (int e = 0; e < 3; ++e)
                for (int k = 0; k < 5; ++k)
                    in[e][k] = 5 * e + k;

            auto as_synthetic = [](int x[3][5], auto loc) {
                return sid::synthetic()
                    .set<property::origin>(sid::host::make_simple_ptr_holder(&x[0][0]))
                    .set<property::strides>(hymap::keys<decltype(loc), unstructured::dim::k>::make_values(5_c, 1_c));
            };
            auto in_s = as_synthetic(in, edge());
            auto out_s = as_synthetic(out, vertex());

            fencil(v2e_conn, 3, 5, out_s, in_s);

            for (int v = 0; v < 3; ++v)
                for (int k = 0; k < 5; ++k) {
                    int nbsum = 0;
                    for (int i = 0; i < 2; ++i) {
                        int nb = v2e_table[v][i];
                        nbsum += in[nb][k];
                    }
                    EXPECT_EQ(out[v][k], nbsum);
                }
        }

    } // namespace
} // namespace gridtools::fn

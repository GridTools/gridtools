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

#include <gridtools/fn/backend2/naive.hpp>
#include <gridtools/fn/unstructured2.hpp>

#include "simple_mesh.hpp"

using namespace gridtools;
using namespace fn;
using namespace literals;
using sid::property;

struct e2v_t {};
struct v2e_t {};

struct zavg_stencil {
    constexpr auto operator()() const {
        return [](auto const &pp, auto const &s) -> tuple<double, double> {
            auto tmp = 0.0;
            tuple_util::for_each(
                [&](auto i) {
                    auto shifted_pp = shift(pp, e2v_t(), i);
                    if (can_deref(shifted_pp))
                        tmp += deref(shifted_pp);
                },
                meta::rename<tuple, meta::make_indices_c<2>>());
            tmp /= 2.0;
            auto ss = deref(s);
            return {tmp * get<0>(ss), tmp * get<1>(ss)};
        };
    }
};

struct nabla_stencil {
    constexpr auto operator()() const {
        return [](auto const &zavg, auto const &sign, auto const &vol) -> tuple<double, double> {
            auto signs = deref(sign);
            auto tmp = tuple(0.0, 0.0);
            tuple_util::for_each(
                [&](auto i) {
                    auto shifted_zavg = shift(zavg, v2e_t(), i);
                    if (can_deref(shifted_zavg)) {
                        tmp = {get<0>(tmp) + get<0>(deref(shifted_zavg)) * get<i.value>(signs),
                            get<1>(tmp) + get<1>(deref(shifted_zavg)) * get<i.value>(signs)};
                    }
                },
                meta::rename<tuple, meta::make_indices_c<6>>());
            auto v = deref(vol);
            return {get<0>(tmp) / v, get<1>(tmp) / v};
        };
    }
};

TEST(unstructured, nabla) {
    using namespace simple_mesh;
    constexpr auto K = 3_c;
    constexpr auto n_v2e = 6_c;

    double pp[n_vertices][K];
    for (auto &ppp : pp)
        for (auto &p : ppp)
            p = rand() % 100;

    std::array<int, n_v2e> sign[n_vertices];
    for (auto &&ss : sign)
        for (auto &s : ss)
            s = rand() % 2 ? 1 : -1;

    double vol[n_vertices];
    for (auto &v : vol)
        v = rand() % 2 + 1;

    tuple<double, double> s[n_edges][K];
    for (auto &ss : s)
        for (auto &sss : ss)
            sss = {rand() % 100, rand() % 100};

    auto zavg = [&](int edge, int k) -> std::array<double, 2> {
        auto tmp = 0.;
        for (auto vertex : e2v[edge])
            tmp += pp[vertex][k];
        tmp /= 2;
        return {tmp * get<0>(s[edge][k]), tmp * get<1>(s[edge][k])};
    };

    auto expected = [&](int vertex, int k) {
        auto res = std::array{0., 0.};
        for (int i = 0; i != 2; ++i) {
            for (int j = 0; j != n_v2e; ++j) {
                auto edge = v2e[vertex][j];
                if (edge == -1)
                    break;
                res[i] += zavg(edge, k)[i] * sign[vertex][j];
            }
            res[i] /= vol[vertex];
        }
        return res;
    };

    tuple<double, double> actual[n_vertices][K] = {};

    auto apply_zavg = [](auto executor, auto &zavg, auto const &pp, auto const &s) {
        executor().arg(zavg).arg(pp).arg(s).assign(0_c, zavg_stencil(), 1_c, 2_c);
    };
    auto apply_nabla = [](auto executor, auto &nabla, auto const &zavg, auto const &sign, auto const &vol) {
        executor().arg(nabla).arg(zavg).arg(sign).arg(vol).assign(0_c, nabla_stencil(), 1_c, 2_c, 3_c);
    };
    auto fencil = [&](auto const &v2e_table,
                      auto const &e2v_table,
                      auto &nabla,
                      auto const &pp,
                      auto const &s,
                      auto const &sign,
                      auto const &vol) {
        using float_t = std::remove_const_t<sid::element_type<decltype(pp)>>;
        auto v2e_conn = connectivity<v2e_t>(v2e_table);
        auto e2v_conn = connectivity<e2v_t>(e2v_table);
        auto edge_domain = unstructured_domain(n_edges, K, e2v_conn);
        auto vertex_domain = unstructured_domain(n_vertices, K, v2e_conn);
        auto edge_backend = make_backend(backend::naive(), edge_domain);
        auto vertex_backend = make_backend(backend::naive(), vertex_domain);
        auto zavg = edge_backend.template make_tmp<tuple<float_t, float_t>>();
        apply_zavg(edge_backend.stencil_executor(), zavg, pp, s);
        apply_nabla(vertex_backend.stencil_executor(), nabla, zavg, sign, vol);
    };

    fencil(&v2e[0], &e2v[0], actual, pp, s, sign, vol);

    for (int h = 0; h < n_vertices; ++h)
        for (int v = 0; v < K; ++v)
            tuple_util::for_each(
                [](auto actual, auto expected) { EXPECT_DOUBLE_EQ(actual, expected); }, actual[h][v], expected(h, v));
}

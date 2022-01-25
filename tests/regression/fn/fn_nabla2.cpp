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

template <class C, int MaxNeighbors, class It>
constexpr auto nb_sum(It const &it) {
    double sum = 0;
    tuple_util::for_each(
        [&](auto offset) {
            auto shifted = shift(it, C(), offset);
            if (can_deref(shifted))
                sum += deref(shifted);
        },
        meta::rename<tuple, meta::make_indices_c<MaxNeighbors>>());
    return sum;
}

template <class C, int MaxNeighbors, class Z, class Sign>
constexpr auto nb_tuple_dot(Z const &z, Sign const &sign) {
    tuple<double, double> res = {0.0, 0.0};
    tuple_util::for_each(
        [&](auto offset) {
            auto shifted_z = shift(z, C(), offset);
            auto shifted_sign = shift(sign, unstructured::dim::neighbor(), offset);
            if (can_deref(shifted_z) && can_deref(shifted_sign)) {
                res = {get<0>(res) + get<0>(deref(shifted_z)) * deref(shifted_sign),
                    get<1>(res) + get<1>(deref(shifted_z)) * deref(shifted_sign)};
            }
        },
        meta::rename<tuple, meta::make_indices_c<MaxNeighbors>>());
    return res;
}

struct e2v_t {};
struct v2e_t {};
struct vertex {};
struct edge {};

struct zavg_stencil {
    constexpr auto operator()() const {
        return [](auto const &pp, auto const &s) -> tuple<double, double> {
            auto tmp = nb_sum<e2v_t, 2>(pp) / 2;
            auto ss = deref(s);
            return {tmp * get<0>(ss), tmp * get<1>(ss)};
        };
    }
};

struct nabla_stencil {
    constexpr auto operator()() const {
        return [](auto const &zavg, auto const &sign, auto const &vol) -> tuple<double, double> {
            auto tmp = nb_tuple_dot<v2e_t, 6>(zavg, sign);
            auto v = deref(vol);
            return {get<0>(tmp) / v, get<1>(tmp) / v};
        };
    }
};

TEST(unstructured, nabla) {
    using namespace simple_mesh;
    constexpr auto K = 3_c;
    constexpr auto n_v2e = 6_c;
    constexpr auto n_e2v = 2_c;

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
                      auto &zavg,
                      auto const &pp,
                      auto const &s,
                      auto const &sign,
                      auto const &vol) {
        auto v2e_conn = connectivity<v2e_t, vertex, edge>(v2e_table, n_v2e);
        auto e2v_conn = connectivity<e2v_t, edge, vertex>(e2v_table, n_e2v);
        auto edge_domain = unstructured_domain<edge>(n_edges, K, e2v_conn);
        auto vertex_domain = unstructured_domain<vertex>(n_vertices, K, v2e_conn);
        auto edge_backend = make_backend(backend::naive(), edge_domain);
        auto vertex_backend = make_backend(backend::naive(), vertex_domain);
        // auto zavg = edge_backend.make_tmp_like(nabla);
        apply_zavg(edge_backend.stencil_executor(), zavg, pp, s);
        apply_nabla(vertex_backend.stencil_executor(), nabla, zavg, sign, vol);
    };
    auto v2e_s = sid::synthetic()
                     .set<property::origin>(sid::host::make_simple_ptr_holder(&v2e[0][0]))
                     .set<property::strides>(hymap::keys<vertex, unstructured::dim::neighbor>::values(n_v2e, 1_c));
    auto e2v_s = sid::synthetic()
                     .set<property::origin>(sid::host::make_simple_ptr_holder(&e2v[0][0]))
                     .set<property::strides>(hymap::keys<edge, unstructured::dim::neighbor>::values(n_e2v, 1_c));
    auto pp_s = sid::synthetic()
                    .set<property::origin>(sid::host::make_simple_ptr_holder(&pp[0][0]))
                    .set<property::strides>(hymap::keys<vertex, unstructured::dim::k>::make_values(K, 1_c));
    auto sign_s =
        sid::synthetic()
            .set<property::origin>(sid::host::make_simple_ptr_holder(&sign[0][0]))
            .set<property::strides>(hymap::keys<vertex, unstructured::dim::neighbor>::make_values(n_v2e, 1_c));
    auto vol_s = sid::synthetic()
                     .set<property::origin>(sid::host::make_simple_ptr_holder(&vol[0]))
                     .set<property::strides>(hymap::keys<vertex>::make_values(1_c));
    auto s_s = sid::synthetic()
                   .set<property::origin>(sid::host::make_simple_ptr_holder(&s[0][0]))
                   .set<property::strides>(hymap::keys<edge, unstructured::dim::k>::make_values(K, 1_c));
    auto actual_s = sid::synthetic()
                        .set<property::origin>(sid::host::make_simple_ptr_holder(&actual[0][0]))
                        .set<property::strides>(hymap::keys<vertex, unstructured::dim::k>::make_values(K, 1_c));

    tuple<double, double> zavgvals[n_edges][K] = {};
    auto zavg_s = sid::synthetic()
                      .set<property::origin>(sid::host::make_simple_ptr_holder(&zavgvals[0][0]))
                      .set<property::strides>(hymap::keys<edge, unstructured::dim::k>::make_values(K, 1_c));
    fencil(v2e_s, e2v_s, actual_s, zavg_s, pp_s, s_s, sign_s, vol_s);

    for (int h = 0; h < n_edges; ++h)
        for (int v = 0; v < K; ++v)
            tuple_util::for_each(
                [](auto actual, auto expected) { EXPECT_DOUBLE_EQ(actual, expected); }, zavgvals[h][v], zavg(h, v));

    for (int h = 0; h < n_vertices; ++h)
        for (int v = 0; v < K; ++v)
            tuple_util::for_each(
                [](auto actual, auto expected) { EXPECT_DOUBLE_EQ(actual, expected); }, actual[h][v], expected(h, v));
}

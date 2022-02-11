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

struct e2v_t {
    static constexpr int max_neighbors = 2;
};
struct v2e_t {
    static constexpr int max_neighbors = 6;
};
struct e2s_t {
    static constexpr int max_neighbors = 2;
};
struct s2e_t {
    static constexpr int max_neighbors = 1;
};
struct v2s_t {
    static constexpr int max_neighbors = 6;
};
struct s2v_t {
    static constexpr int max_neighbors = 1;
};

struct e2s_table {};

constexpr tuple<int, int> neighbor_table_neighbors(e2s_table, int index) { return {2 * index, 2 * index + 1}; }

struct s2e_table {};

constexpr tuple<int> neighbor_table_neighbors(s2e_table, int index) { return {index / 2}; }

template <class V2ETable, class E2VTable>
struct v2s_table {
    V2ETable m_v2e_table;
    E2VTable m_e2v_table;
};

template <class V2ETable, class E2VTable>
constexpr auto neighbor_table_neighbors(v2s_table<V2ETable, E2VTable> const &v2s, int vertex) {
    auto edges = neighbor_table::neighbors(v2s.m_v2e_table, vertex);
    return tuple_util::transform(
        [&](int edge) {
            if (edge == -1)
                return -1;
            auto vertices = neighbor_table::neighbors(v2s.m_e2v_table, edge);
            auto sparse = neighbor_table::neighbors(e2s_table{}, edge);
            if (get<0>(vertices) == vertex) {
                return get<0>(sparse);
            } else {
                assert(get<1>(vertices) == vertex);
                return get<1>(sparse);
            }
        },
        edges);
}

template <class E2VTable>
struct s2v_table {
    E2VTable m_e2v_table;
};

template <class E2VTable>
constexpr auto neighbor_table_neighbors(s2v_table<E2VTable> const &s2v, int sparse) {
    auto edges = neighbor_table::neighbors(s2e_table{}, sparse);
    return tuple_util::transform(
        [&](int edge) {
            if (edge == -1)
                return -1;
            auto sparse2 = neighbor_table::neighbors(e2s_table{}, edge);
            auto vertices = neighbor_table::neighbors(s2v.m_e2v_table, edge);
            if (get<0>(sparse2) == sparse) {
                return get<0>(vertices);
            } else {
                assert(get<1>(sparse2) == sparse);
                return get<1>(vertices);
            }
        },
        edges);
}

struct zavg_stencil {
    constexpr auto operator()() const {
        return [](auto const &pp, auto const &s) -> tuple<double, double> {
            auto tmp = reduce(e2v_t(), std::plus(), 0.0, pp) / 2;
            auto ss = deref(s);
            return {tmp * get<0>(ss), tmp * get<1>(ss)};
        };
    }
};

struct nabla_stencil {
    constexpr auto operator()() const {
        return [](auto const &zavg, auto const &sign, auto const &vol) -> tuple<double, double> {
            auto tmp = tuple(0.0, 0.0);
            tuple_util::for_each(
                [&](auto i) {
                    auto shifted_zavg = shift(zavg, v2e_t(), i);
                    auto shifted_sign = shift(sign, v2s_t(), i);
                    if (can_deref(shifted_zavg) && can_deref(shifted_sign)) {
                        tmp = {get<0>(tmp) + get<0>(deref(shifted_zavg)) * deref(shifted_sign),
                            get<1>(tmp) + get<1>(deref(shifted_zavg)) * deref(shifted_sign)};
                    }
                },
                meta::rename<tuple, meta::make_indices_c<v2e_t::max_neighbors>>());
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

    int sign[2 * n_edges];
    for (auto &&s : sign)
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
                auto sparse = 2 * edge + (e2v[edge][0] == vertex ? 0 : 1);
                res[i] += zavg(edge, k)[i] * sign[sparse];
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
                      auto const &v2s_table,
                      auto &nabla,
                      auto const &pp,
                      auto const &s,
                      auto const &sign,
                      auto const &vol) {
        auto v2e_conn = connectivity<v2e_t>(v2e_table);
        auto e2v_conn = connectivity<e2v_t>(e2v_table);
        auto v2s_conn = connectivity<v2s_t>(v2s_table);
        auto edge_domain = unstructured_domain(n_edges, K, e2v_conn);
        auto vertex_domain = unstructured_domain(n_vertices, K, v2e_conn, v2s_conn);
        auto edge_backend = make_backend(backend::naive(), edge_domain);
        auto vertex_backend = make_backend(backend::naive(), vertex_domain);
        auto zavg = edge_backend.make_tmp_like(nabla);
        apply_zavg(edge_backend.stencil_executor(), zavg, pp, s);
        apply_nabla(vertex_backend.stencil_executor(), nabla, zavg, sign, vol);
    };
    v2s_table<decltype(&v2e[0]), decltype(&e2v[0])> v2s{v2e, e2v};

    fencil(v2e, e2v, v2s, actual, pp, s, sign, vol);

    for (int h = 0; h < n_vertices; ++h)
        for (int v = 0; v < K; ++v)
            tuple_util::for_each(
                [](auto actual, auto expected) { EXPECT_DOUBLE_EQ(actual, expected); }, actual[h][v], expected(h, v));
}

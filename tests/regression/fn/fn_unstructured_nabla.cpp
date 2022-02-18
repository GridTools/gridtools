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

#include <gridtools/fn/unstructured2.hpp>

#include <fn_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace fn;
    using namespace literals;

    struct zavg_stencil {
        constexpr auto operator()() const {
            return [](auto const &pp, auto const &s) {
                std::decay_t<decltype(deref(pp))> tmp = 0;
                tuple_util::for_each(
                    [&](auto i) {
                        auto shifted_pp = shift(pp, e2v(), i);
                        if (can_deref(shifted_pp))
                            tmp += deref(shifted_pp);
                    },
                    meta::rename<tuple, meta::make_indices_c<2>>());
                tmp /= 2.0;
                auto ss = deref(s);
                return tuple(tmp * get<0>(ss), tmp * get<1>(ss));
            };
        }
    };

    struct nabla_stencil {
        constexpr auto operator()() const {
            return [](auto const &zavg, auto const &sign, auto const &vol) {
                using float_t = std::decay_t<decltype(deref(vol))>;
                auto signs = deref(sign);
                tuple<float_t, float_t> tmp(0, 0);
                tuple_util::for_each(
                    [&](auto i) {
                        auto shifted_zavg = shift(zavg, v2e(), i);
                        if (can_deref(shifted_zavg)) {
                            get<0>(tmp) += get<0>(deref(shifted_zavg)) * get<i.value>(signs);
                            get<1>(tmp) += get<1>(deref(shifted_zavg)) * get<i.value>(signs);
                        }
                    },
                    meta::rename<tuple, meta::make_indices_c<6>>());
                auto v = deref(vol);
                return tuple(get<0>(tmp) / v, get<1>(tmp) / v);
            };
        }
    };

    GT_REGRESSION_TEST(fn_unstructured_nabla, test_environment<>, fn_backend_t) {
        using float_t = typename TypeParam::float_t;

        auto mesh = TypeParam::fn_unstructured_mesh();

        auto v2e_table = mesh.v2e_table();
        auto e2v_table = mesh.e2v_table();

        auto v2e_view = v2e_table->const_host_view();
        auto e2v_view = e2v_table->const_host_view();

        auto pp = [](int vertex, int k) -> float_t { return (vertex + k) % 19; };
        auto sign = [](int vertex) -> array<float_t, 6> {
            return {0, 1, float_t(vertex % 2), 1, float_t((vertex + 1) % 2), 0};
        };
        auto vol = [](int vertex) -> float_t { return vertex % 13 + 1; };
        auto s = [](int edge, int k) -> tuple<float_t, float_t> { return {(edge + k) % 17, (edge + k) % 7}; };
        auto zavg = [&](int edge, int k) -> tuple<float_t, float_t> {
            float_t tmp = 0;
            for (int neighbor = 0; neighbor < 2; ++neighbor)
                tmp += pp(e2v_view(edge, neighbor), k);
            tmp /= 2;
            return {tmp * get<0>(s(edge, k)), tmp * get<1>(s(edge, k))};
        };
        auto expected = [&](int vertex, int k) {
            auto res = tuple(float_t(0), float_t(0));
            for (int neighbor = 0; neighbor < 6; ++neighbor) {
                int edge = v2e_view(vertex, neighbor);
                if (edge != -1) {
                    get<0>(res) += get<0>(zavg(edge, k)) * sign(vertex)[neighbor];
                    get<1>(res) += get<1>(zavg(edge, k)) * sign(vertex)[neighbor];
                }
            }
            get<0>(res) /= vol(vertex);
            get<1>(res) /= vol(vertex);
            return res;
        };

        auto apply_zavg = [](auto executor, auto &zavg, auto const &pp, auto const &s) {
            executor().arg(zavg).arg(pp).arg(s).assign(0_c, zavg_stencil(), 1_c, 2_c);
        };
        auto apply_nabla = [](auto executor, auto &nabla, auto const &zavg, auto const &sign, auto const &vol) {
            executor().arg(nabla).arg(zavg).arg(sign).arg(vol).assign(0_c, nabla_stencil(), 1_c, 2_c, 3_c);
        };
        auto fencil = [&](int nvertices,
                          int nedges,
                          int nlevels,
                          auto const &v2e_table,
                          auto const &e2v_table,
                          auto &nabla,
                          auto const &pp,
                          auto const &s,
                          auto const &sign,
                          auto const &vol) {
            auto v2e_conn = connectivity<v2e>(v2e_table);
            auto e2v_conn = connectivity<e2v>(e2v_table);
            auto edge_domain = unstructured_domain(nedges, nlevels, e2v_conn);
            auto vertex_domain = unstructured_domain(nvertices, nlevels, v2e_conn);
            auto edge_backend = make_backend(backend::naive(), edge_domain);
            auto vertex_backend = make_backend(backend::naive(), vertex_domain);
            auto zavg = edge_backend.make_tmp_like(nabla);
            apply_zavg(edge_backend.stencil_executor(), zavg, pp, s);
            apply_nabla(vertex_backend.stencil_executor(), nabla, zavg, sign, vol);
        };

        auto comp = [&,
                        v2e_table = reinterpret_cast<array<int, 6> const *>(v2e_table->get_const_target_ptr()),
                        e2v_table = reinterpret_cast<array<int, 2> const *>(e2v_table->get_const_target_ptr()),
                        pp = mesh.make_const_storage(pp, mesh.nvertices(), mesh.nlevels()),
                        sign = mesh.template make_const_storage<array<float_t, 6>>(sign, mesh.nvertices()),
                        vol = mesh.make_const_storage(vol, mesh.nvertices()),
                        s = mesh.template make_const_storage<tuple<float_t, float_t>>(
                            s, mesh.nedges(), mesh.nlevels())](auto &nabla) {
            fencil(mesh.nvertices(), mesh.nedges(), mesh.nlevels(), v2e_table, e2v_table, nabla, pp, s, sign, vol);
        };
        {
            auto nabla = mesh.template make_storage<tuple<float_t, float_t>>(mesh.nvertices(), mesh.nlevels());
            comp(nabla);
            TypeParam::verify(expected, nabla);
            TypeParam::benchmark("fn_unstructured_nabla", std::bind(comp, nabla));
        }
        {
            auto nabla0 = mesh.make_storage(mesh.nvertices(), mesh.nlevels());
            auto nabla1 = mesh.make_storage(mesh.nvertices(), mesh.nlevels());
            auto nabla =
                sid::composite::keys<integral_constant<int, 0>, integral_constant<int, 1>>::make_values(nabla0, nabla1);
            comp(nabla);
            TypeParam::verify([&](int vertex, int k) { return get<0>(expected(vertex, k)); }, nabla0);
            TypeParam::verify([&](int vertex, int k) { return get<1>(expected(vertex, k)); }, nabla1);
        }
    }
} // namespace

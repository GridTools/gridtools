/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/icosahedral.hpp>

#include <type_traits>

#include <backend_select.hpp>
#include <test_environment.hpp>

#include "curl_functors.hpp"
#include "operators_repository.hpp"

namespace {
    using namespace gridtools;
    using namespace ico_operators;

    template <class Env>
    const auto eq = [](auto lhs, auto rhs) {
        return expect_with_threshold(lhs, rhs, std::is_same<typename Env::float_t, float>::value ? 1e-4 : 1e-9);
    };

    GT_REGRESSION_TEST(curl_weights, icosahedral_test_environment<2>, backend_t) {
        auto spec = [](auto reciprocal, auto edge_length, auto in_edges, auto out) {
            GT_DECLARE_ICO_TMP((array<typename TypeParam::float_t, 6>), vertices, weights);
            return execute_parallel()
                .ij_cached(weights)
                .stage(curl_prep_functor(), reciprocal, edge_length, weights)
                .stage(curl_functor_weights(), in_edges, weights, out);
        };
        operators_repository repo = {TypeParam::d(0), TypeParam::d(1)};
        auto out = TypeParam ::icosahedral_make_storage(vertices());
        run(spec,
            backend_t(),
            TypeParam ::make_grid(),
            TypeParam ::icosahedral_make_storage(vertices(), repo.dual_area_reciprocal),
            TypeParam ::icosahedral_make_storage(edges(), repo.dual_edge_length),
            TypeParam ::icosahedral_make_storage(edges(), repo.u),
            out);
        TypeParam::verify(repo.curl_u, out, eq<TypeParam>);
    }

    GT_REGRESSION_TEST(curl_flow_convention, icosahedral_test_environment<2>, backend_t) {
        operators_repository repo = {TypeParam::d(0), TypeParam::d(1)};
        auto out = TypeParam ::icosahedral_make_storage(vertices());
        run_single_stage(curl_functor_flow_convention(),
            backend_t(),
            TypeParam ::make_grid(),
            TypeParam ::icosahedral_make_storage(edges(), repo.u),
            TypeParam ::icosahedral_make_storage(vertices(), repo.dual_area_reciprocal),
            TypeParam ::icosahedral_make_storage(edges(), repo.dual_edge_length),
            out);
        TypeParam ::verify(repo.curl_u, out, eq<TypeParam>);
    }
} // namespace

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

#include <gridtools/fn/cartesian2.hpp>
#include <gridtools/fn/unstructured2.hpp>

#include <fn_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace fn;
    using namespace literals;

    struct copy_stencil {
        GT_FUNCTION constexpr auto operator()() const {
            return [](auto const &in) { return deref(in); };
        }
    };

    constexpr inline auto in = [](auto... indices) { return (... + indices); };

    constexpr inline auto apply_copy = [](auto &&executor, auto &out, auto const &in) {
        executor().arg(out).arg(in).assign(0_c, copy_stencil(), 1_c);
    };

    GT_REGRESSION_TEST(fn_cartesian_copy, test_environment<>, fn_backend_t) {
        auto out = TypeParam::make_storage();
        auto fencil = [&](auto const &sizes, auto &out, auto const &in) {
            auto domain = cartesian_domain(sizes);
            auto backend = make_backend(fn_backend_t(), domain);
            apply_copy(backend.stencil_executor(), out, in);
        };

        auto comp = [&, in = TypeParam::make_const_storage(in)] { fencil(TypeParam::fn_cartesian_sizes(), out, in); };
        comp();
        TypeParam::verify(in, out);
        TypeParam::benchmark("fn_cartesian_copy", comp);
    }

    GT_REGRESSION_TEST(fn_unstructured_copy, test_environment<>, fn_backend_t) {
        auto fencil = [&](int nvertices, int nlevels, auto &out, auto const &in) {
            auto domain = unstructured_domain(nvertices, nlevels);
            auto backend = make_backend(fn_backend_t(), domain);
            apply_copy(backend.stencil_executor(), out, in);
        };

        auto mesh = TypeParam::fn_unstructured_mesh();
        auto out = mesh.make_storage(mesh.nvertices(), mesh.nlevels());
        auto comp = [&, in = mesh.make_const_storage(in, mesh.nvertices(), mesh.nlevels())] {
            fencil(mesh.nvertices(), mesh.nlevels(), out, in);
        };
        comp();
        TypeParam::verify(in, out);
        TypeParam::benchmark("fn_unstructured_copy", comp);
    }
} // namespace

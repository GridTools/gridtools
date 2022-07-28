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

#include <gridtools/fn/cartesian.hpp>
#include <gridtools/fn/unstructured.hpp>

#include <fn_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace fn;
    using namespace literals;

    struct forward_scan : fwd {
        static GT_FUNCTION constexpr auto body() {
            return scan_pass(
                [](auto acc, auto const &field) {
                    return acc + deref(field);
                },
                host_device::identity());
        }
    };

    constexpr inline auto field_function = [](auto...) { return 1; };
    constexpr inline auto expected = [](auto...) { return 1; };

    constexpr inline auto tridiagonal_solve = [](auto executor, auto const &field, auto &x) {
        using float_t = sid::element_type<decltype(field)>;
        executor()
            .arg(field)
            .arg(x)
            .assign(1_c, forward_scan(), float_t(0), 0_c)
            .execute();
    };

    GT_REGRESSION_TEST(fn_vertical_indirection, vertical_test_environment<>, fn_backend_t) {
        using float_t = typename TypeParam::float_t;

        auto fencil = [&](int nvertices, int nlevels, auto const &field, auto &x) {
            auto be = fn_backend_t();
            auto domain = unstructured_domain({nvertices, nlevels}, {});
            auto backend = make_backend(be, domain);
            auto alloc = tmp_allocator(be);
            tridiagonal_solve(backend.vertical_executor(), field, x);
        };

        auto mesh = TypeParam::fn_unstructured_mesh();
        auto output_storage = mesh.make_storage(mesh.nvertices(), mesh.nlevels());
        const auto field_storage = mesh.make_const_storage(field_function, mesh.nvertices(), mesh.nlevels());
        fencil(mesh.nvertices(), mesh.nlevels(), field_storage, output_storage);
        TypeParam::verify(expected, output_storage);
    }
} // namespace

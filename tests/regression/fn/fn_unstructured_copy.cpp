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
    using namespace unstructured;
    using namespace literals;

    struct copy_stencil {
        GT_FUNCTION constexpr auto operator()() const {
            return [](auto const &in) { return deref(in); };
        }
    };

    GT_REGRESSION_TEST(fn_unstructured_copy, test_environment<>, fn_backend_t) {
        auto in = [](int i, int j) { return i + j; };
        auto builder = storage::builder<typename TypeParam::storage_traits_t>.dimensions(TypeParam::fn_unstructured_nvertices(), TypeParam::fn_unstructured_nlevels()).template type<typename TypeParam::float_t>();
        auto out = builder.build();

        auto apply_copy = [](auto executor, auto &out, auto const &in) {
            executor().arg(out).arg(in).assign(0_c, copy_stencil(), 1_c);
        };

        auto fencil = [&](int nvertices, int nlevels, auto &out, auto const &in) {
            auto domain = unstructured_domain(nvertices, nlevels);
            auto backend = make_backend(fn_backend_t(), domain);
            apply_copy(backend.stencil_executor(), out, in);
        };

        auto comp = [&, in = builder.initializer(in).build()] {
            fencil(TypeParam::fn_unstructured_nvertices(), TypeParam::fn_unstructured_nlevels(), out, in);
        };
        comp();
        TypeParam::verify(in, out);
        TypeParam::benchmark("fn_unstructured_copy", comp);
    }
} // namespace

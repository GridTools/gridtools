/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cstddef>
#include <gtest/gtest.h>

#include <gridtools/fn/cartesian.hpp>
#include <gridtools/fn/unstructured.hpp>

#include <fn_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace fn;
    using namespace literals;

    struct forward_scan {
        auto operator()() const {
            return [](auto const &field, auto const &vertical_offset) {
                const auto shifted = shift(field, unstructured::dim::vertical{}, deref(vertical_offset));
                return deref(field) - deref(shifted);
            };
        }
    };

    constexpr auto input_function = []([[maybe_unused]] auto horizontal_idx, auto vertical_idx) -> float {
        const float x = float(vertical_idx);
        const float xp = x + 0.5f;
        const float y = 0.5f * xp * xp;
        return y;
    };
    constexpr auto vertical_offset_function = []([[maybe_unused]] auto horizontal_idx, auto vertical_idx) -> int {
        return vertical_idx > 0 ? -1 : 0;
    };
    constexpr auto expected = []([[maybe_unused]] auto horizontal_idx, auto vertical_idx) { return vertical_idx; };

    constexpr auto vertical_indirection =
        [](auto executor, auto const &input, auto &output, auto const &vertical_offsets) {
            using float_t = sid::element_type<decltype(input)>;
            executor()
                .arg(input)
                .arg(vertical_offsets)
                .arg(output)
                .assign(2_c, forward_scan(), 0_c, 1_c)
                .execute();
        };

    GT_REGRESSION_TEST(fn_vertical_indirection, test_environment<>, fn_backend_t) {
        using float_t = typename TypeParam::float_t;

        auto fencil = [&](int nvertices, int nlevels, auto const &field, auto &output, auto &vertical_offsets) {
            auto be = fn_backend_t();
            auto domain = unstructured_domain({nvertices, nlevels}, tuple{0, 0});
            auto backend = make_backend(be, domain);
            auto alloc = tmp_allocator(be);
            vertical_indirection(backend.stencil_executor(), field, output, vertical_offsets);
        };

        auto mesh = TypeParam::fn_unstructured_mesh();
        auto output_storage = mesh.make_storage(mesh.nvertices(), mesh.nlevels());
        const auto input_storage = mesh.make_const_storage(input_function, mesh.nvertices(), mesh.nlevels());
        const auto vertical_offset_storage =
            mesh.template make_const_storage<int>(vertical_offset_function, mesh.nvertices(), mesh.nlevels());
        fencil(mesh.nvertices(), mesh.nlevels(), input_storage, output_storage, vertical_offset_storage);
        TypeParam::verify(expected, output_storage);
    }
} // namespace

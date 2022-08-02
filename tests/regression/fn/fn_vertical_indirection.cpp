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

    struct derivative_stencil {
        auto operator()() const {
            return [](auto const &field, auto const &offset) {
                const auto shifted = shift(field, unstructured::dim::vertical{}, deref(offset));
                return deref(field) - deref(shifted);
            };
        }
    };

    constexpr auto parabolic = []([[maybe_unused]] auto horizontal_idx, auto vertical_idx) -> float {
        const float x = float(vertical_idx);
        const float xp = x + 0.5f;
        const float y = 0.5f * xp * xp;
        return y;
    };
    constexpr auto offsets = []([[maybe_unused]] auto horizontal_idx, auto vertical_idx) -> int {
        return vertical_idx > 0 ? -1 : 0;
    };
    constexpr auto linear = []([[maybe_unused]] auto horizontal_idx, auto vertical_idx) { return vertical_idx; };

    constexpr auto compute_derivative = [](auto executor, auto const &input, auto const &offsets, auto &output) {
        using float_t = sid::element_type<decltype(input)>;
        executor().arg(input).arg(offsets).arg(output).assign(2_c, derivative_stencil(), 0_c, 1_c).execute();
    };

    GT_REGRESSION_TEST(fn_vertical_indirection, test_environment<>, fn_backend_t) {
        using float_t = typename TypeParam::float_t;

        auto fencil = [](int nvertices, int nlevels, auto const &field, auto &offets, auto &output) {
            auto be = fn_backend_t();
            auto domain = unstructured_domain({nvertices, nlevels}, tuple{0, 0});
            auto backend = make_backend(be, domain);
            auto alloc = tmp_allocator(be);
            compute_derivative(backend.stencil_executor(), field, offets, output);
        };

        auto mesh = TypeParam::fn_unstructured_mesh();
        const auto input_offsets = mesh.template make_const_storage<int>(offsets, mesh.nvertices(), mesh.nlevels());
        const auto input_parabola = mesh.make_const_storage(parabolic, mesh.nvertices(), mesh.nlevels());
        auto output_derivative = mesh.make_storage(mesh.nvertices(), mesh.nlevels());
        fencil(mesh.nvertices(), mesh.nlevels(), input_parabola, input_offsets, output_derivative);
        TypeParam::verify(linear, output_derivative);
    }
} // namespace

/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <array>
#include <cstdlib>
#include <utility>

#include <gtest/gtest.h>

#include "../common/halo_descriptor.hpp"
#include "../stencil_composition/axis.hpp"
#include "../stencil_composition/grid.hpp"
#include "backend_select.hpp"
#include "verifier.hpp"

namespace gridtools {
    template <size_t Halo = 0, class Axis = axis<1>, class = std::make_index_sequence<Axis::n_intervals>>
    class grid_fixture;

    template <size_t Halo, class Axis, size_t... Is>
    class grid_fixture<Halo, Axis, std::index_sequence<Is...>> : public virtual testing::Test {
        std::array<size_t, 2 + sizeof...(Is)> m_dims;

        template <size_t>
        using just_size_t = size_t;

      public:
        grid_fixture(size_t d0, size_t d1, just_size_t<Is>... ds) : m_dims{d0, d1, ds...} {}

        auto &d(size_t i) { return m_dims[i]; }
        auto const &d(size_t i) const { return m_dims[i]; }

        auto k_size() const { return make_grid().k_size(typename Axis::full_interval()); }

        auto make_grid() const {
            auto halo_desc = [](auto d) { return halo_descriptor(Halo, Halo, Halo, d - Halo - 1, d); };
            return ::gridtools::make_grid(halo_desc(m_dims[0]), halo_desc(m_dims[1]), Axis(m_dims[2 + Is]...));
        }

        template <class Expected, class Actual, class EqualTo = default_equal_to<typename Actual::element_type>>
        void verify(Expected const &expected, Actual const &actual, EqualTo equal_to = {}) const {
            std::array<std::array<size_t, 2>, Actual::element_type::ndims> halos = {{{Halo, Halo}, {Halo, Halo}}};
            EXPECT_TRUE(verify_data_store(expected, actual, halos, equal_to));
        }
    };
} // namespace gridtools

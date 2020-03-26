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

#include <gridtools/common/halo_descriptor.hpp>
#include <gridtools/stencil_composition/frontend/axis.hpp>
#include <gridtools/stencil_composition/frontend/make_grid.hpp>

#include "verifier.hpp"

namespace gridtools {

    template <uint_t... Is>
    struct static_domain {
        static auto d(size_t i) { return ((uint_t[]){Is...})[i]; }
        static size_t steps() { return 0; }
        static bool needs_verification() { return true; }

        static void flush_cache() {}
    };

    template <class Domain, size_t Halo = 0, class Axis = axis<1>, class = std::make_index_sequence<Axis::n_intervals>>
    struct grid_fixture;

    template <class Domain, size_t Halo, class Axis, size_t... Is>
    struct grid_fixture<Domain, Halo, Axis, std::index_sequence<Is...>> {
        static auto d(size_t i) { return Domain::d(i); }

        static auto k_size() { return make_grid().k_size(typename Axis::full_interval()); }

        static auto make_grid() {
            auto halo_desc = [](auto d) { return halo_descriptor(Halo, Halo, Halo, d - Halo - 1, d); };
            return ::gridtools::make_grid(halo_desc(d(0)), halo_desc(d(1)), Axis(d(2 + Is)...));
        }

        template <class Expected, class Actual, class EqualTo = default_equal_to<typename Actual::element_type>>
        static void verify(Expected const &expected, Actual const &actual, EqualTo equal_to = {}) {
            if (!Domain::needs_verification())
                return;
            std::array<std::array<size_t, 2>, Actual::element_type::ndims> halos = {{{Halo, Halo}, {Halo, Halo}}};
            EXPECT_TRUE(verify_data_store(expected, actual, halos, equal_to));
        }
    };
} // namespace gridtools

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

#include <cassert>
#include <type_traits>
#include <utility>
#include <vector>

#include <gridtools/common/array.hpp>
#include <gridtools/common/defs.hpp>
#include <gridtools/stencil_composition/icosahedral_grids/icosahedral_topology.hpp>
#include <gridtools/stencil_composition/location_type.hpp>

namespace gridtools {
    namespace _impl {
        struct neighbour {
            int_t i, c, j, k;
            template <class Fun>
            auto call(Fun &&fun) const {
                return std::forward<Fun>(fun)(i, c, j, k);
            }
        };

        template <class FromLocation, class ToLocation>
        std::vector<array<int_t, 4>> get_offsets(
            uint_t, std::integral_constant<uint_t, FromLocation::n_colors::value>) {
            assert(false);
            return {};
        }

        template <class FromLocation,
            class ToLocation,
            uint_t C = 0,
            std::enable_if_t<(C < FromLocation::n_colors::value), int> = 0>
        std::vector<array<int_t, 4>> get_offsets(uint_t c, std::integral_constant<uint_t, C> = {}) {
            if (c > C) {
                return get_offsets<FromLocation, ToLocation>(c, std::integral_constant<uint_t, C + 1>{});
            }
            auto offsets = connectivity<FromLocation, ToLocation, C>::offsets();
            return std::vector<array<int_t, 4>>(offsets.begin(), offsets.end());
        }
    } // namespace _impl

    template <class FromLocation, class ToLocation>
    std::vector<_impl::neighbour> neighbours_of(int_t i, int_t c, int_t j, int_t k) {
        assert(c >= 0);
        assert(c < FromLocation::n_colors::value);
        std::vector<_impl::neighbour> res;
        for (auto &item : _impl::get_offsets<FromLocation, ToLocation>(c))
            res.push_back({i + item[0], c + item[1], j + item[2], k + item[3]});
        return res;
    };
} // namespace gridtools

/*
 * GridTools
 *
 * Copyright (c) 2014-2022, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <gridtools/sid/concept.hpp>
#include <type_traits>

namespace gridtools::fn::sid_neighbor_table {
    namespace sid_neighbor_table_impl_ {
        template <std::size_t MaxNumNeighbors, class Sid>
        struct sid_neighbor_table {
            Sid sid;
        };

        template <std::size_t MaxNumNeighbors, class Sid>
        auto neighbor_table_neighbors(sid_neighbor_table<MaxNumNeighbors, Sid> const &table, size_t index) {
            using element_type = sid::element_type<Sid>;
            const auto ptr = sid_get_origin(table.sid);
            const auto lower_bounds = sid_get_lower_bounds(table.sid);
            const auto upper_bounds = sid_get_upper_bounds(table.sid);
            const auto strides = sid_get_strides(table.sid);
            std::array<element_type, MaxNumNeighbors> neighbors;
            std::fill(std::begin(neighbors), std::end(neighbors), element_type{-1});
            return neighbors;
        }

        template <std::size_t MaxNumNeighbors, class Sid>
        auto as_neighbor_table(Sid sid) {
            return sid_neighbor_table<MaxNumNeighbors, Sid>{std::move(sid)};
        }
    } // namespace sid_neighbor_table_impl_

    using sid_neighbor_table_impl_::as_neighbor_table;

} // namespace gridtools::fn::sid_neighbor_table
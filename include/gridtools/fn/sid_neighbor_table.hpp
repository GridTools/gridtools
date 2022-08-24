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

#include <cstddef>
#include <type_traits>

#include "../common/array.hpp"
#include "../fn/unstructured.hpp"
#include "../sid/concept.hpp"

namespace gridtools::fn::sid_neighbor_table {
    namespace sid_neighbor_table_impl_ {
        template <class IndexDimension, class NeighborDimension, std::size_t MaxNumNeighbors, class Sid>
        struct sid_neighbor_table {
            Sid sid;
        };

        template <class IndexDimension, class NeighborDimension, std::size_t MaxNumNeighbors, class Sid>
        auto neighbor_table_neighbors(
            sid_neighbor_table<IndexDimension, NeighborDimension, MaxNumNeighbors, Sid> const &table,
            std::size_t index) {
            using element_type = sid::element_type<Sid>;

            auto ptr = sid::get_origin(table.sid)();
            const auto strides = sid::get_strides(table.sid);

            gridtools::array<element_type, MaxNumNeighbors> neighbors;

            sid::shift(ptr, sid::get_stride<IndexDimension>(strides), index);
            for (std::size_t elementIdx = 0; elementIdx < MaxNumNeighbors; ++elementIdx) {
                neighbors[elementIdx] = *ptr;
                sid::shift(ptr, sid::get_stride<NeighborDimension>(strides), 1);
            }
            return neighbors;
        }

        template <class IndexDimension, class NeighborDimension, int32_t MaxNumNeighbors, class Sid>
        auto as_neighbor_table(Sid &&sid) {
            static_assert(gridtools::tuple_util::size<decltype(sid::get_strides(std::declval<Sid>()))>::value == 2,
                "Neighbor tables must have exactly two dimensions: the index dimension and the neighbor dimension");
            static_assert(!std::is_same_v<IndexDimension, NeighborDimension>,
                "The index dimension and the neighbor dimension must be different.");

            return sid_neighbor_table<IndexDimension, NeighborDimension, MaxNumNeighbors, Sid>{std::forward<Sid>(sid)};
        }
    } // namespace sid_neighbor_table_impl_

    using sid_neighbor_table_impl_::as_neighbor_table;

} // namespace gridtools::fn::sid_neighbor_table
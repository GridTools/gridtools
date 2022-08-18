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

#include <cstdint>
#include <gridtools/common/array.hpp>
#include <gridtools/fn/unstructured.hpp>
#include <gridtools/sid/concept.hpp>
#include <type_traits>

namespace gridtools::fn::sid_neighbor_table {
    namespace sid_neighbor_table_impl_ {
        template <class IndexDimension, class NeighborDimension, int32_t MaxNumNeighbors, class Sid>
        struct sid_neighbor_table {
            Sid sid;
        };

        template <class IndexDimension, class NeighborDimension, int32_t MaxNumNeighbors, class Sid>
        auto neighbor_table_neighbors(
            sid_neighbor_table<IndexDimension, NeighborDimension, MaxNumNeighbors, Sid> const &table, size_t index) {
            using element_type = sid::element_type<Sid>;

            const auto ptr = sid_get_origin(table.sid);
            const auto strides = sid_get_strides(table.sid);

            const auto index_stride = at_key<IndexDimension>(strides);
            const auto neighbour_stride = at_key<NeighborDimension>(strides);

            gridtools::array<element_type, MaxNumNeighbors> neighbors;
            for (int32_t elementIdx = 0; elementIdx < MaxNumNeighbors; ++elementIdx) {
                const auto element_ptr = ptr + index * index_stride + elementIdx * neighbour_stride;
                neighbors[elementIdx] = *element_ptr();
            }
            return neighbors;
        }

        template <class IndexDimension, class NeighborDimension, int32_t MaxNumNeighbors, class Sid>
        auto as_neighbor_table(Sid sid) {
            static_assert(gridtools::tuple_util::size<decltype(sid_get_strides(std::declval<Sid>()))>::value == 2,
                "Neighbor tables must have exactly two dimensions: the index dimension and the neighbor dimension");
            static_assert(!std::is_same_v<IndexDimension, NeighborDimension>,
                "The index dimension and the neighbor dimension must be different.");

            return sid_neighbor_table<IndexDimension, NeighborDimension, MaxNumNeighbors, Sid>{std::move(sid)};
        }
    } // namespace sid_neighbor_table_impl_

    using sid_neighbor_table_impl_::as_neighbor_table;

} // namespace gridtools::fn::sid_neighbor_table
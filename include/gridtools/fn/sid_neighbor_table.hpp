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
        template <class IndexDimension,
            class NeighborDimension,
            std::size_t MaxNumNeighbors,
            class PtrHolder,
            class Strides>
        struct sid_neighbor_table {
            PtrHolder origin;
            Strides strides;

            GT_FUNCTION friend auto neighbor_table_neighbors(sid_neighbor_table const &table, int index) {
                auto ptr = table.origin();
                using element_type = std::remove_reference_t<decltype(*ptr)>;

                gridtools::array<element_type, MaxNumNeighbors> neighbors;

                sid::shift(ptr, sid::get_stride<IndexDimension>(table.strides), index);
                for (std::size_t element_idx = 0; element_idx < MaxNumNeighbors; ++element_idx) {
                    neighbors[element_idx] = *ptr;
                    sid::shift(ptr, sid::get_stride<NeighborDimension>(table.strides), 1);
                }
                return neighbors;
            }
        };

        template <class IndexDimension, class NeighborDimension, int32_t MaxNumNeighbors, class Sid>
        auto as_neighbor_table(Sid &&sid) -> sid_neighbor_table<IndexDimension,
            NeighborDimension,
            MaxNumNeighbors,
            sid::ptr_holder_type<Sid>,
            sid::strides_type<Sid>> {

            static_assert(gridtools::tuple_util::size<decltype(sid::get_strides(std::declval<Sid>()))>::value == 2,
                "Neighbor tables must have exactly two dimensions: the index dimension and the neighbor dimension");
            static_assert(!std::is_same_v<IndexDimension, NeighborDimension>,
                "The index dimension and the neighbor dimension must be different.");

            const auto origin = sid::get_origin(sid);
            const auto strides = sid::get_strides(sid);

            return {origin, strides};
        }
    } // namespace sid_neighbor_table_impl_

    using sid_neighbor_table_impl_::as_neighbor_table;

} // namespace gridtools::fn::sid_neighbor_table
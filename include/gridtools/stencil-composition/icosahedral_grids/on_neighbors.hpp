/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../../meta/type_traits.hpp"
#include "../is_accessor.hpp"
#include "../location_type.hpp"

namespace gridtools {

    /**
     *  This struct is the one holding the function to apply when iterating on neighbors
     */
    template <typename ValueType, typename DstLocationType, typename ReductionFunction, typename... Accessors>
    struct on_neighbors {
        ReductionFunction m_function;
        ValueType m_value;
    };

    template <typename Reduction, typename ValueType, typename... Accessors>
    constexpr GT_FUNCTION on_neighbors<ValueType, enumtype::edges, Reduction, Accessors...> on_edges(
        Reduction function, ValueType initial, Accessors...) {
        GT_STATIC_ASSERT(conjunction<is_accessor<Accessors>...>::value, "'on_edges' arguments should be accessors");
        GT_STATIC_ASSERT((conjunction<std::is_same<typename Accessors::location_type, enumtype::edges>...>::value),
            "'on_edges' arguments should be accessors with the 'edges' location type.");
        return {function, initial};
    }

    template <typename Reduction, typename ValueType, typename... Accessors>
    constexpr GT_FUNCTION on_neighbors<ValueType, enumtype::cells, Reduction, Accessors...> on_cells(
        Reduction function, ValueType initial, Accessors...) {
        GT_STATIC_ASSERT(conjunction<is_accessor<Accessors>...>::value, "'on_cells' arguments should be accessors");
        GT_STATIC_ASSERT((conjunction<std::is_same<typename Accessors::location_type, enumtype::cells>...>::value),
            "'on_cells' arguments should be accessors with the 'cells' location type.");
        return {function, initial};
    }

    template <typename Reduction, typename ValueType, typename... Accessors>
    constexpr GT_FUNCTION on_neighbors<ValueType, enumtype::vertices, Reduction, Accessors...> on_vertices(
        Reduction function, ValueType initial, Accessors...) {
        GT_STATIC_ASSERT(conjunction<is_accessor<Accessors>...>::value, "'on_vertices' arguments should be accessors");
        GT_STATIC_ASSERT((conjunction<std::is_same<typename Accessors::location_type, enumtype::vertices>...>::value),
            "'on_vertices' arguments should be accessors with the 'vertices' location type.");
        return {function, initial};
    }
} // namespace gridtools

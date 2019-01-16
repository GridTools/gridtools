/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
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
        GRIDTOOLS_STATIC_ASSERT(
            conjunction<is_accessor<Accessors>...>::value, "'on_edges' arguments should be accessors");
        GRIDTOOLS_STATIC_ASSERT(
            (conjunction<std::is_same<typename Accessors::location_type, enumtype::edges>...>::value),
            "'on_edges' arguments should be accessors with the 'edges' location type.");
        return {function, initial};
    }

    template <typename Reduction, typename ValueType, typename... Accessors>
    constexpr GT_FUNCTION on_neighbors<ValueType, enumtype::cells, Reduction, Accessors...> on_cells(
        Reduction function, ValueType initial, Accessors...) {
        GRIDTOOLS_STATIC_ASSERT(
            conjunction<is_accessor<Accessors>...>::value, "'on_cells' arguments should be accessors");
        GRIDTOOLS_STATIC_ASSERT(
            (conjunction<std::is_same<typename Accessors::location_type, enumtype::cells>...>::value),
            "'on_cells' arguments should be accessors with the 'cells' location type.");
        return {function, initial};
    }

    template <typename Reduction, typename ValueType, typename... Accessors>
    constexpr GT_FUNCTION on_neighbors<ValueType, enumtype::vertices, Reduction, Accessors...> on_vertices(
        Reduction function, ValueType initial, Accessors...) {
        GRIDTOOLS_STATIC_ASSERT(
            conjunction<is_accessor<Accessors>...>::value, "'on_vertices' arguments should be accessors");
        GRIDTOOLS_STATIC_ASSERT(
            (conjunction<std::is_same<typename Accessors::location_type, enumtype::vertices>...>::value),
            "'on_vertices' arguments should be accessors with the 'vertices' location type.");
        return {function, initial};
    }
} // namespace gridtools

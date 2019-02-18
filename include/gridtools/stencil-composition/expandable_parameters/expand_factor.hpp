/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <type_traits>

/**@file
   Expand factor for expandable parameters encoding the unrolling factor
   for the loop over the expandable parameters.
*/

namespace gridtools {
    /** @brief factor determining the length of the "chunks" in an expandable parameters list
        \tparam Tile The unrlolling factor
     */
    template <size_t Value>
    struct expand_factor : std::integral_constant<size_t, Value> {};

    template <class>
    struct is_expand_factor : std::false_type {};

    template <size_t Value>
    struct is_expand_factor<expand_factor<Value>> : std::true_type {};
} // namespace gridtools

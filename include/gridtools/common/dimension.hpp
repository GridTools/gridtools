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

#include <type_traits>

#include "defs.hpp"
#include "host_device.hpp"

namespace gridtools {
    /** \ingroup common
        @{
        \defgroup dimension Dimension
        @{
    */

    /**
       @brief The following struct defines one specific component of a field
       It contains a direction (compile time constant, specifying the ID of the component),
       and a value (runtime value, which is storing the offset in the given direction).
    */
    template <ushort_t Coordinate>
    struct dimension {
        GT_STATIC_ASSERT(Coordinate != 0, "The coordinate values passed to the accessor start from 1");

        GT_FUNCTION constexpr dimension() : value(0) {}

        template <typename IntType>
        GT_FUNCTION constexpr dimension(IntType val) : value{(int_t)val} {}

        dimension(dimension const &) = default;

        static constexpr ushort_t index = Coordinate;
        int_t value;
    };

    template <typename T>
    struct is_dimension : std::false_type {};

    template <ushort_t Id>
    struct is_dimension<dimension<Id>> : std::true_type {};

    /** @} */
    /** @} */
} // namespace gridtools

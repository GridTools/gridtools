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

#include "array.hpp"
#include "defs.hpp"
#include "generic_metafunctions/utility.hpp"
#include "tuple_util.hpp"

namespace gridtools {
    /** \ingroup array
        @{
    */

    /** \brief Facility to make an array given a variadic list of
        values.

        Facility to make an array given a variadic list of
        values.An explicit template argument can be used to force the
        value type of the array. The list of values passed to the
        function must have a common type or be covertible to the
        explici value type if that is specified. The size of the array
        is the length of the list of values.

        \tparam ForceType Value type of the resulting array (optional)
        \param values List of values to put in the array. The length of the list set the size of the array.
     */
    template <typename ForceType = void, typename... Types>
    constexpr GT_FUNCTION auto make_array(Types &&... values)
        GT_AUTO_RETURN((tuple_util::host_device::make<array, ForceType>(const_expr::forward<Types>(values)...)));

    /** @} */
} // namespace gridtools

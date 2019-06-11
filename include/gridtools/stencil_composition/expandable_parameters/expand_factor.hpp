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

/**@file
   Expand factor for expandable parameters encoding the unrolling factor
   for the loop over the expandable parameters.
*/

namespace gridtools {
    template <size_t Value>
    using expand_factor = std::integral_constant<size_t, Value>;
} // namespace gridtools

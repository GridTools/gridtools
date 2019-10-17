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

#include "../../common/defs.hpp"

namespace gridtools {

    /** \ingroup storage
     * @{
     */

    /**
     * @brief This struct is used to pass alignment information to storage info types. Alignment is in terms of number
     * of elements.
     * @tparam alignment value
     */
    template <uint_t N>
    struct alignment {
        static_assert(N > 0, "Alignment value must be greater than 0");
        static constexpr uint_t value = N;
    };

    template <typename T>
    struct is_alignment : std::false_type {};

    template <uint_t N>
    struct is_alignment<alignment<N>> : std::true_type {};

    /**
     * @}
     */
} // namespace gridtools

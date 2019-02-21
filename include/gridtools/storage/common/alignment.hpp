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

#include "../../common/array.hpp"
#include "../../common/gt_assert.hpp"
#include "definitions.hpp"
#include "storage_info_metafunctions.hpp"

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
        GT_STATIC_ASSERT(N > 0, "Alignment value must be greater than 0");
        const static uint_t value = N;
    };

    template <typename T>
    struct is_alignment : boost::mpl::false_ {};

    template <uint_t N>
    struct is_alignment<alignment<N>> : boost::mpl::true_ {};

    /**
     * @}
     */
} // namespace gridtools

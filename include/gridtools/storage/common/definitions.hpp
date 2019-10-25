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

namespace gridtools {
    /** \ingroup storage
     * @{
     */

    enum class ownership { external_gpu, external_cpu };
    enum class access_mode { read_write = 0, read_only = 1 };

    template <access_mode Mode, class T = void>
    using access_mode_type =
        std::integral_constant<access_mode, std::is_const<T>::value ? access_mode::read_only : Mode>;

    using access_mode_read_write_t = access_mode_type<access_mode::read_write>;
    using access_mode_read_only_t = access_mode_type<access_mode::read_only>;

    template <access_mode Mode, class T>
    using apply_access_mode = std::conditional_t<Mode == access_mode::read_only, T const, T>;

    /**
     * @}
     */
} // namespace gridtools

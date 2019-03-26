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
#include "../../common/timer/timer_traits.hpp"
#include "../backend_traits_fwd.hpp"

/**@file
 * @brief type definitions and structures specific for the X86 backend
 */
namespace gridtools {
    /**Traits struct, containing the types which are specific for the x86 backend*/
    template <>
    struct backend_traits<target::x86> {
        /**
         * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
         */
        using mss_fuse_esfs_strategy = std::false_type;

        using performance_meter_t = typename timer_traits<target::x86>::timer_type;
    };

} // namespace gridtools

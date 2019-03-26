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
@brief type definitions and structures specific for the CUDA backend*/
namespace gridtools {
    /** @brief traits struct defining the types which are specific to the CUDA backend*/
    template <>
    struct backend_traits<target::cuda> {

        /**
         * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
         */
        using mss_fuse_esfs_strategy = std::true_type;

        using performance_meter_t = typename timer_traits<target::cuda>::timer_type;
    };

} // namespace gridtools

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

#include <utility>

#include "../../common/functional.hpp"
#include "../../common/timer/timer_traits.hpp"
#include "../backend_traits_fwd.hpp"
#include "../mss_functor.hpp"
#include "../structured_grids/backend_mc/execute_kernel_functor_mc.hpp"

/**@file
@brief type definitions and structures specific for the Mic backend
*/
namespace gridtools {

    /**Traits struct, containing the types which are specific for the mc backend*/
    template <>
    struct backend_traits<target::mc> {
        /**
         * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
         */
        using mss_fuse_esfs_strategy = std::true_type;

        using performance_meter_t = typename timer_traits<target::mc>::timer_type;
    };
} // namespace gridtools

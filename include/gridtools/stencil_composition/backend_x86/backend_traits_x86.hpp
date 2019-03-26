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
#include "strategy_x86.hpp"

/**@file
 * @brief type definitions and structures specific for the X86 backend
 */
namespace gridtools {
    /**Traits struct, containing the types which are specific for the x86 backend*/
    template <>
    struct backend_traits_from_id<target::x86> {
        /**
         * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
         * and sequentially executes all the functors in the mss
         * @tparam RunFunctorArgs run functor arguments
         */
        template <typename RunFunctorArgs>
        struct mss_loop {
            typedef typename RunFunctorArgs::backend_ids_t backend_ids_t;

            GT_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);
            template <typename LocalDomain, typename Grid>
            static void run(
                LocalDomain const &local_domain, Grid const &grid, const execution_info_x86 &execution_info) {
                GT_STATIC_ASSERT((is_local_domain<LocalDomain>::value), GT_INTERNAL_ERROR);
                GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

                // each strategy executes a different high level loop for a mss
                strategy_from_id_x86<typename backend_ids_t::strategy_id_t>::template mss_loop<
                    RunFunctorArgs>::template run(local_domain, grid, execution_info);
            }
        };

        /**
         * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
         */
        typedef std::false_type mss_fuse_esfs_strategy;

        // metafunction that contains the strategy from id metafunction corresponding to this backend
        template <typename BackendIds>
        struct select_strategy {
            GT_STATIC_ASSERT((is_backend_ids<BackendIds>::value), GT_INTERNAL_ERROR);
            typedef strategy_from_id_x86<typename BackendIds::strategy_id_t> type;
        };

        using performance_meter_t = typename timer_traits<target::x86>::timer_type;
    };

} // namespace gridtools

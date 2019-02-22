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
#include "../structured_grids/backend_mc/execute_kernel_functor_mc.hpp"
#include "../structured_grids/backend_mc/strategy_mc.hpp"

/**@file
@brief type definitions and structures specific for the Mic backend
*/
namespace gridtools {
    /**Traits struct, containing the types which are specific for the mc backend*/
    template <>
    struct backend_traits_from_id<target::mc> {

        /** This is the functor used to generate view instances. According to the given storage an appropriate view is
         * returned. When using the Host backend we return host view instances.
         */
        struct make_view_f {
            template <typename S, typename SI>
            auto operator()(data_store<S, SI> const &src) const GT_AUTO_RETURN(make_host_view(src));
        };

        template <uint_t Id>
        struct once_per_block {
            template <typename Left, typename Right>
            GT_FUNCTION static void assign(Left &l, Right const &r) {
                l = r;
            }
        };

        /**
         * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
         * and sequentially executes all the functors in the mss
         * @tparam RunFunctorArgs run functor arguments
         */
        template <typename RunFunctorArgs>
        struct mss_loop {
            typedef typename RunFunctorArgs::backend_ids_t backend_ids_t;

            GT_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);
            template <typename LocalDomain, typename Grid, typename ExecutionInfo>
            GT_FUNCTION static void run(
                LocalDomain const &local_domain, Grid const &grid, const ExecutionInfo &execution_info) {
                GT_STATIC_ASSERT((is_local_domain<LocalDomain>::value), GT_INTERNAL_ERROR);
                GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

                strgrid::execute_kernel_functor_mc<RunFunctorArgs>(local_domain, grid)(execution_info);
            }
        };

        /**
         * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
         */
        using mss_fuse_esfs_strategy = std::true_type;

        // metafunction that contains the strategy from id metafunction corresponding to this backend
        template <typename BackendIds>
        struct select_strategy {
            GT_STATIC_ASSERT((is_backend_ids<BackendIds>::value), GT_INTERNAL_ERROR);
            typedef strategy_from_id_mc<typename BackendIds::strategy_id_t> type;
        };

        using performance_meter_t = typename timer_traits<target::mc>::timer_type;
    };

} // namespace gridtools

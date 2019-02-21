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

#include <cuda_runtime.h>

#include "../../common/defs.hpp"
#include "../../common/timer/timer_traits.hpp"
#include "../../storage/data_store.hpp"

#include "../backend_traits_fwd.hpp"
#include "../grid_traits_fwd.hpp"
#include "execute_kernel_functor_cuda.hpp"
#include "strategy_cuda.hpp"

/**@file
@brief type definitions and structures specific for the CUDA backend*/
namespace gridtools {

    /** @brief traits struct defining the types which are specific to the CUDA backend*/
    template <>
    struct backend_traits_from_id<target::cuda> {

        /** This is the functor used to generate view instances. According to the given storage
           an appropriate view is returned. When using the CUDA backend we return device view instances.
        */
        struct make_view_f {
            template <typename S, typename SI>
            auto operator()(data_store<S, SI> const &src) const GT_AUTO_RETURN(make_device_view(src));
        };

        /**
           @brief assigns the two given values using the given thread Id whithin the block
        */
        template <uint_t Id>
        struct once_per_block {
            template <typename Left, typename Right>
            GT_FUNCTION static void assign(Left &l, Right const &r) {
                assert(blockDim.z == 1);
                if (Id % (blockDim.x * blockDim.y) == threadIdx.y * blockDim.x + threadIdx.x)
                    l = r;
            }
        };

        /**
         * @brief main execution of a mss.
         * @tparam RunFunctorArgs run functor arguments
         */
        template <typename RunFunctorArgs>
        struct mss_loop {
            typedef typename RunFunctorArgs::backend_ids_t backend_ids_t;

            GT_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);
            template <typename LocalDomain, typename Grid, typename ExecutionInfo>
            static void run(LocalDomain &local_domain, const Grid &grid, ExecutionInfo &&) {
                typedef typename kernel_functor_executor<backend_ids_t, RunFunctorArgs>::type kernel_functor_executor_t;
                kernel_functor_executor_t(local_domain, grid)();
            }
        };

        /**
         * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
         */
        typedef std::true_type mss_fuse_esfs_strategy;

        // metafunction that contains the strategy from id metafunction corresponding to this backend
        template <typename BackendIds>
        struct select_strategy {
            GT_STATIC_ASSERT((is_backend_ids<BackendIds>::value), GT_INTERNAL_ERROR);
            typedef strategy_from_id_cuda<typename BackendIds::strategy_id_t> type;
        };

        using performance_meter_t = typename timer_traits<target::cuda>::timer_type;
    };

} // namespace gridtools

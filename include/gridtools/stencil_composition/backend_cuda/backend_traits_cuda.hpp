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

#include "../backend_traits_fwd.hpp"
#include "../mss_functor.hpp"
#include "execute_kernel_functor_cuda.hpp"

/**@file
@brief type definitions and structures specific for the CUDA backend*/
namespace gridtools {

    /** @brief traits struct defining the types which are specific to the CUDA backend*/
    template <>
    struct backend_traits_from_id<target::cuda> {
        /**
           @brief assigns the two given values using the given thread Id whithin the block
        */
        template <uint_t Id>
        struct once_per_block {
            template <typename Left, typename Right>
            GT_FUNCTION static void assign(Left &GT_RESTRICT l, Right const &GT_RESTRICT r) {
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
                execute_kernel_functor_cuda<RunFunctorArgs>(local_domain, grid)();
            }
        };

        /**
         * @brief struct holding backend-specific runtime information about stencil execution.
         * Empty for the CUDA backend.
         */
        struct execution_info_cuda {};

        /**
         * @brief loops over all blocks and execute sequentially all mss functors for each block
         * @tparam MssComponents a meta array with the mss components of all MSS
         * @tparam BackendIds backend ids type
         */
        template <typename MssComponents, typename BackendIds>
        struct fused_mss_loop {
            GT_STATIC_ASSERT((is_sequence_of<MssComponents, is_mss_components>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT((is_backend_ids<BackendIds>::value), GT_INTERNAL_ERROR);

            template <typename LocalDomainListArray, typename Grid>
            static void run(LocalDomainListArray const &local_domain_lists, const Grid &grid) {
                GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

                host::for_each<GT_META_CALL(meta::make_indices, boost::mpl::size<MssComponents>)>(
                    mss_functor<MssComponents, Grid, LocalDomainListArray, BackendIds, execution_info_cuda>(
                        local_domain_lists, grid, {}));
            }
        };

        /**
         * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
         */
        typedef std::true_type mss_fuse_esfs_strategy;

        using performance_meter_t = typename timer_traits<target::cuda>::timer_type;
    };

} // namespace gridtools

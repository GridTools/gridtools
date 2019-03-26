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

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../meta.hpp"
#include "../backend_ids.hpp"
#include "../grid.hpp"
#include "../mss_components.hpp"

namespace gridtools {

    template <typename MssComponentsArray,
        typename Grid,
        typename MssLocalDomainArray,
        typename BackendIds,
        typename ExecutionInfo>
    struct mss_functor;

    template <class>
    struct strategy_from_id_cuda;

    /**
     * @brief struct holding backend-specific runtime information about stencil execution.
     * Empty for the CUDA backend.
     */
    struct execution_info_cuda {};

    /**
       @brief specialization for the \ref strategy::block strategy
       Empty as not used in the CUDA backend
    */
    template <>
    struct strategy_from_id_cuda<strategy::block> {
        /**
         * @brief loops over all blocks and execute sequentially all mss functors for each block
         * @tparam MssComponents a meta array with the mss components of all MSS
         * @tparam BackendIds backend ids type
         */
        template <typename MssComponents, typename BackendIds>
        struct fused_mss_loop {
            GT_STATIC_ASSERT((meta::all_of<is_mss_components, MssComponents>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT(is_backend_ids<BackendIds>::value, GT_INTERNAL_ERROR);

            template <typename LocalDomainListArray, typename Grid>
            static void run(LocalDomainListArray const &local_domain_lists, const Grid &grid) {
                GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

                host::for_each<GT_META_CALL(meta::make_indices_for, MssComponents)>(
                    mss_functor<MssComponents, Grid, LocalDomainListArray, BackendIds, execution_info_cuda>(
                        local_domain_lists, grid, {}));
            }
        };
    };

} // namespace gridtools

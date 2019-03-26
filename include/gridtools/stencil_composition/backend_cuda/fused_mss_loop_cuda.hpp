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

#include <boost/mpl/size.hpp>

#include "../mss_functor.hpp"

/**@file
 * @brief fused mss loop implementations for the cuda backend
 */
namespace gridtools {
    /**
     * @brief struct holding backend-specific runtime information about stencil execution.
     * Empty for the CUDA backend.
     */
    struct execution_info_cuda {};

    /**
     * @brief loops over all blocks and execute sequentially all mss functors for each block
     * @tparam MssComponents a meta array with the mss components of all MSS
     */
    template <class MssComponents, class LocalDomainListArray, class Grid>
    static void fused_mss_loop(
        target::cuda const &backend_target, LocalDomainListArray const &local_domain_lists, const Grid &grid) {
        GT_STATIC_ASSERT((is_sequence_of<MssComponents, is_mss_components>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

        host::for_each<GT_META_CALL(meta::make_indices, boost::mpl::size<MssComponents>)>(
            make_mss_functor<MssComponents>(backend_target, local_domain_lists, grid, execution_info_cuda{}));
    }
} // namespace gridtools

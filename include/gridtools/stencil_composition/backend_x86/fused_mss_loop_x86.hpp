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

#include "../../meta.hpp"
#include "../mss_functor.hpp"

/**@file
 * @brief fused mss loop implementations for the x86 backend
 */
namespace gridtools {
    /**
     * @brief struct holding backend-specific runtime information about stencil execution.
     */
    struct execution_info_x86 {
        int_t bi, bj;
    };

    /**
     * @brief loops over all blocks and execute sequentially all mss functors for each block
     * @tparam MssComponents a meta array with the mss components of all MSS
     */
    template <class MssComponents, class LocalDomainListArray, class Grid>
    void fused_mss_loop(backend::x86, LocalDomainListArray const &local_domain_lists, const Grid &grid) {
        GT_STATIC_ASSERT((meta::all_of<is_mss_components, MssComponents>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
        int_t n = grid.i_size() - 1;
        int_t m = grid.j_size() - 1;

        int_t NBI = n / block_i_size(backend::x86{});
        int_t NBJ = m / block_j_size(backend::x86{});

#pragma omp parallel
        {
#pragma omp for nowait
            for (int_t bi = 0; bi <= NBI; ++bi) {
                for (int_t bj = 0; bj <= NBJ; ++bj) {
                    run_mss_functors<MssComponents>(
                        backend::x86{}, local_domain_lists, grid, execution_info_x86{bi, bj});
                }
            }
        }
    }

    /**
     * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
     */
    constexpr std::false_type mss_fuse_esfs(backend::x86) { return {}; }
} // namespace gridtools

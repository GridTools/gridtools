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
 * @brief fused mss loop implementations for the x86 backend
 */
namespace gridtools {
    /**
     * @brief struct holding backend-specific runtime information about stencil execution.
     */
    struct execution_info_x86 {
        uint_t bi, bj;
    };

    /**
     * @brief loops over all blocks and execute sequentially all mss functors for each block
     * @tparam MssComponents a meta array with the mss components of all MSS
     */
    template <class MssComponents, class LocalDomainListArray, class Grid>
    GT_FORCE_INLINE static void fused_mss_loop(
        target::x86 const &backend_target, LocalDomainListArray const &local_domain_lists, const Grid &grid) {
        GT_STATIC_ASSERT((is_sequence_of<MssComponents, is_mss_components>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);
        using iter_range = boost::mpl::range_c<uint_t, 0, boost::mpl::size<MssComponents>::type::value>;

        uint_t n = grid.i_high_bound() - grid.i_low_bound();
        uint_t m = grid.j_high_bound() - grid.j_low_bound();

        uint_t NBI = n / block_i_size(backend_target);
        uint_t NBJ = m / block_j_size(backend_target);

#pragma omp parallel
        {
#pragma omp for nowait
            for (uint_t bi = 0; bi <= NBI; ++bi) {
                for (uint_t bj = 0; bj <= NBJ; ++bj) {
                    host::for_each<GT_META_CALL(meta::make_indices, boost::mpl::size<MssComponents>)>(
                        make_mss_functor<MssComponents>(
                            backend_target, local_domain_lists, grid, execution_info_x86{bi, bj}));
                }
            }
        }
    }

    /**
     * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
     */
    constexpr std::false_type mss_fuse_esfs(target::x86) { return {}; }
} // namespace gridtools

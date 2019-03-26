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

#include "./execute_kernel_functor_x86.hpp"

/**@file
 * @brief mss loop implementations for the x86 backend
 */
namespace gridtools {

    /**
     * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
     * and sequentially executes all the functors in the mss
     * @tparam RunFunctorArgs run functor arguments
     */
    template <class RunFunctorArgs, class LocalDomain, class Grid, class ExecutionInfo>
    static void mss_loop(target::x86 const &backend_target,
        LocalDomain const &local_domain,
        Grid const &grid,
        const ExecutionInfo &execution_info) {
        GT_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_local_domain<LocalDomain>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

        auto block_size_f = [](uint_t total, uint_t block_size, uint_t block_no) {
            auto n = (total + block_size - 1) / block_size;
            return block_no == n - 1 ? total - block_no * block_size : block_size;
        };
        auto total_i = grid.i_high_bound() - grid.i_low_bound() + 1;
        auto total_j = grid.j_high_bound() - grid.j_low_bound() + 1;

        execute_kernel_functor_x86<RunFunctorArgs>{local_domain,
            grid,
            block_size_f(total_i, block_i_size(backend_target), execution_info.bi),
            block_size_f(total_j, block_j_size(backend_target), execution_info.bj),
            execution_info.bi,
            execution_info.bj}();
    }

} // namespace gridtools

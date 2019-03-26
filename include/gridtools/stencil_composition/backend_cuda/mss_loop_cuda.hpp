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

#include "../structured_grids/backend_cuda/execute_kernel_functor_cuda.hpp"

/**@file
 * @brief mss loop implementations for the cuda backend
 */
namespace gridtools {

    /**
     * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
     * and sequentially executes all the functors in the mss
     * @tparam RunFunctorArgs run functor arguments
     */
    template <class RunFunctorArgs, class LocalDomain, class Grid, class ExecutionInfo>
    static void mss_loop(
        target::cuda const &backend_target, LocalDomain const &local_domain, Grid const &grid, const ExecutionInfo &) {
        GT_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_local_domain<LocalDomain>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

        execute_kernel_functor_cuda<RunFunctorArgs>(local_domain, grid)();
    }

} // namespace gridtools

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
 * @brief mss loop implementations for the x86 backend
 */
namespace gridtools {
    /**
     * @brief loops over all blocks and execute sequentially all mss functors for each block
     * @tparam MssComponents a meta array with the mss components of all MSS
     */
    template <class MssComponents, class LocalDomains, class Grid>
    void fused_mss_loop(backend::naive, LocalDomains const &local_domains, Grid const &grid) {
        run_mss_functors<MssComponents>(backend::naive{}, local_domains, grid, 0);
    }

    /**
     * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
     */
    constexpr std::true_type mss_fuse_esfs(backend::naive) { return {}; }
} // namespace gridtools

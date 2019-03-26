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
#include "../../common/gt_math.hpp"
#include "../../common/pair.hpp"
#include "../backend_ids.hpp"
#include "../basic_token_execution.hpp"
#include "../execution_types.hpp"

namespace gridtools {
    /**
     * get_k_interval specialization for parallel execution policy. The full k-axis is split into equally-sized block
     * sub-intervals and assigned to the blocks in z-direction. Each block iterates over all computation intervals and
     * calculates the subinterval which intersects with the block sub-interval.
     *
     * Example with with an axis with four intervals, that is distributed among 2 blocks:
     *
     * Computation intervals   block sub-intervals
     *                           with two blocks
     *
     *                                       B1   B2
     *    0 ---------           0 --------- ---         Block B1 calculates the complete intervals I1 and I2, and parts
     *          |                     |    1 :          of I3. It does not calculate anything for I4.
     *          |  I1             B1  |      :
     *          |                     |      :          1. iteration: get_k_interval(...) = [0, 4]
     *          |                     |      :          2. iteration: get_k_interval(...) = [5, 7]
     *    5    ---                    |     ---         3. iteration: get_k_interval(...) = [8, 9]
     *          |                     |    2 :          4. iteration: get_k_interval(...) = [18, 9] (= no calculation)
     *          | I2                  |      :
     *    8    ---                    |     ---
     *          |                     |    3 :
     *          | I3           10    ---    ---  ---    Block B2 calculates parts of the interval I3, and the complete
     *          |                     |         3 :     interval I4. It does not calculate anything for I1 and I2.
     *          |                 B2  |           :
     *          |                     |           :     1. iteration: get_k_interval(...) = [10, 4] (= no calculation)
     *          |                     |           :     2. iteration: get_k_interval(...) = [10, 7] (= no calculation)
     *          |                     |           :     3. iteration: get_k_interval(...) = [10, 17]
     *          |                     |           :     4. iteration: get_k_interval(...) = [18, 20]
     *          |                     |           :
     *   18    ---                    |          ---
     *          | I4                  |         4 :
     *   20 ---------          20 ---------      ---
     */
    template <class FromLevel, class ToLevel, uint_t BlockSize, class Grid>
    GT_FUNCTION pair<int, int> get_k_interval(
        backend_ids<target::cuda>, execute::parallel_block<BlockSize>, Grid const &grid) {
        return make_pair(math::max(blockIdx.z * BlockSize, grid.template value_at<FromLevel>()),
            math::min((blockIdx.z + 1) * BlockSize - 1, grid.template value_at<ToLevel>()));
    }

    template <class FromLevel, class ToLevel, class ExecutionEngine, class Grid>
    GT_FUNCTION pair<int, int> get_k_interval(backend_ids<target::cuda>, ExecutionEngine, Grid const &grid) {
        return make_pair(grid.template value_at<FromLevel>(), grid.template value_at<ToLevel>());
    }
} // namespace gridtools

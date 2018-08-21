/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#pragma once

#include "../../common/defs.hpp"
#include "../../common/pair.hpp"
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
    template <class FromLevel, class ToLevel, class GridBackend, class Strategy, uint_t BlockSize, class Grid>
    GT_FUNCTION pair<int, int> get_k_interval(backend_ids<platform::cuda, GridBackend, Strategy>,
        enumtype::execute<enumtype::parallel, BlockSize>,
        Grid const &grid) {
        return make_pair(math::max(blockIdx.z * BlockSize, grid.template value_at<FromLevel>()),
            math::min((blockIdx.z + 1) * BlockSize - 1, grid.template value_at<ToLevel>()));
    }

    template <class FromLevel, class ToLevel, class GridBackend, class Strategy, class ExecutionEngine, class Grid>
    GT_FUNCTION pair<int, int> get_k_interval(
        backend_ids<platform::cuda, GridBackend, Strategy>, ExecutionEngine, Grid const &grid) {
        return make_pair(grid.template value_at<FromLevel>(), grid.template value_at<ToLevel>());
    }
} // namespace gridtools

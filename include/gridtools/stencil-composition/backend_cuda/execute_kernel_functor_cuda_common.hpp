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
#include "../../common/host_device.hpp"
#include "../execution_types.hpp"

namespace gridtools {
    namespace impl_ {

        /**
         * @brief One block in z for forward/backward execution.
         */
        template <typename ExecutionType>
        struct blocks_required_z {
            GT_FUNCTION static uint_t get(uint_t /*nz*/) { return 1; }
        };

        /**
         * @brief Compute number of blocks in z direction with `BlockSize` fused levels for parallel execution policy.
         */
        template <uint_t BlockSize>
        struct blocks_required_z<execute::parallel_block<BlockSize>> {
            GT_FUNCTION static uint_t get(uint_t nz) { return (nz + BlockSize - 1) / BlockSize; }
        };

        template <typename ExecutionType>
        struct compute_kblock {
            template <typename from_t, typename GridType>
            GT_FUNCTION_DEVICE static int_t get(GridType const &grid) {
                // Note: We subtract grid.k_min() here as it will be added again in
                // it_domain.initialize()
                return grid.template value_at<from_t>() - grid.k_min();
            }
        };

        template <uint_t BlockSize>
        struct compute_kblock<execute::parallel_block<BlockSize>> {
            template <typename from_t, typename GridType>
            GT_FUNCTION_DEVICE static int_t get(GridType const &grid) {
                // Note: We subtract grid.k_min() here as it will be added again in
                // it_domain.initialize()
                return max(blockIdx.z * BlockSize, grid.template value_at<from_t>()) - grid.k_min();
            }
        };
    } // namespace impl_
} // namespace gridtools

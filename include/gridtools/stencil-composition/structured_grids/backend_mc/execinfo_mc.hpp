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

#include "../../../common/defs.hpp"
#include "../../../common/host_device.hpp"

namespace gridtools {

    /**
     *  @brief Execution info class for MC backend.
     *  Used for stencils that are executed serially along the k-axis.
     */
    struct execinfo_block_kserial_mc {
        int_t i_first;      /** First index in block along i-axis. */
        int_t j_first;      /** First index in block along j-axis. */
        int_t i_block_size; /** Size of block along i-axis. */
        int_t j_block_size; /** Size of block along j-axis. */
    };

    /**
     *  @brief Execution info class for MC backend.
     *  Used for stencils that are executed in parallel along the k-axis.
     */
    struct execinfo_block_kparallel_mc {
        int_t i_first;      /** First index in block along i-axis. */
        int_t j_first;      /** First index in block along j-axis. */
        int_t k;            /** Position along k-axis. */
        int_t i_block_size; /** Size of block along i-axis. */
        int_t j_block_size; /** Size of block along j-axis. */
    };

    /**
     * @brief Helper class for block handling.
     */
    class execinfo_mc {
      public:
        using block_kserial_t = execinfo_block_kserial_mc;
        using block_kparallel_t = execinfo_block_kparallel_mc;

        template <class Grid>
        GT_FUNCTION execinfo_mc(const Grid &grid)
            : m_i_grid_size(grid.i_high_bound() - grid.i_low_bound() + 1),
              m_j_grid_size(grid.j_high_bound() - grid.j_low_bound() + 1), m_i_low_bound(grid.i_low_bound()),
              m_j_low_bound(grid.j_low_bound()) {
            const int_t threads = omp_get_max_threads();

            // if domain is large enough (relative to the number of threads),
            // we split only along j-axis (for prefetching reasons)
            // for smaller domains we also split along i-axis
            m_j_block_size = (m_j_grid_size + threads - 1) / threads;
            m_j_blocks = (m_j_grid_size + m_j_block_size - 1) / m_j_block_size;
            const int_t max_i_blocks = threads / m_j_blocks;
            m_i_block_size = (m_i_grid_size + max_i_blocks - 1) / max_i_blocks;
            m_i_blocks = (m_i_grid_size + m_i_block_size - 1) / m_i_block_size;

            assert(m_i_block_size > 0 && m_j_block_size > 0);
        }

        /**
         * @brief Computes the effective (clamped) block size and position for k-serial stencils.
         *
         * @param i_block_index Block index along i-axis.
         * @param j_block_index Block index along j-axis.
         *
         * @return An execution info instance with the computed properties.
         */
        GT_FUNCTION block_kserial_t block(int_t i_block_index, int_t j_block_index) const {
            return block_kserial_t{block_start(i_block_index, m_i_block_size, m_i_low_bound),
                block_start(j_block_index, m_j_block_size, m_j_low_bound),
                clamped_block_size(m_i_grid_size, i_block_index, m_i_block_size, m_i_blocks),
                clamped_block_size(m_j_grid_size, j_block_index, m_j_block_size, m_j_blocks)};
        }

        /**
         * @brief Computes the effective (clamped) block size and position for k-parallel stencils.
         *
         * @param i_block_index Block index along i-axis.
         * @param j_block_index Block index along j-axis.
         * @param k Index along k-axis.
         *
         * @return An execution info instance with the computed properties.
         */
        GT_FUNCTION block_kparallel_t block(int_t i_block_index, int_t j_block_index, int_t k) const {
            return block_kparallel_t{block_start(i_block_index, m_i_block_size, m_i_low_bound),
                block_start(j_block_index, m_j_block_size, m_j_low_bound),
                k,
                clamped_block_size(m_i_grid_size, i_block_index, m_i_block_size, m_i_blocks),
                clamped_block_size(m_j_grid_size, j_block_index, m_j_block_size, m_j_blocks)};
        }

        /** @brief Number of blocks along i-axis. */
        GT_FUNCTION int_t i_blocks() const { return m_i_blocks; }
        /** @brief Number of blocks along j-axis. */
        GT_FUNCTION int_t j_blocks() const { return m_j_blocks; }

        /** @brief Unclamped block size along i-axis. */
        GT_FUNCTION int_t i_block_size() const { return m_i_block_size; }
        /** @brief Unclamped block size along j-axis. */
        GT_FUNCTION int_t j_block_size() const { return m_j_block_size; }

      private:
        GT_FUNCTION static int_t block_start(int_t block_index, int_t block_size, int_t offset) {
            return block_index * block_size + offset;
        }

        GT_FUNCTION static int_t clamped_block_size(
            int_t grid_size, int_t block_index, int_t block_size, int_t blocks) {
            return (block_index == blocks - 1) ? grid_size - block_index * block_size : block_size;
        }

        int_t m_i_grid_size, m_j_grid_size;
        int_t m_i_low_bound, m_j_low_bound;
        int_t m_i_block_size, m_j_block_size;
        int_t m_i_blocks, m_j_blocks;
    };

} // namespace gridtools

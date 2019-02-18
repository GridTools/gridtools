/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
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

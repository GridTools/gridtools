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

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../execution_types.hpp"
#include "../grid.hpp"
#include "basic_token_execution_cuda.hpp"
#include "iterate_domain_cuda.hpp"
#include "launch_kernel.hpp"

namespace gridtools {
    namespace cuda {
        namespace fused_mss_loop_cuda_impl_ {
            template <class Grid>
            GT_FUNCTION_DEVICE auto compute_kblock(execute::forward, Grid const &grid) {
                return grid.k_min();
            };

            template <class Grid>
            GT_FUNCTION_DEVICE auto compute_kblock(execute::backward, Grid const &grid) {
                return grid.k_max();
            };

            template <class ExecutionType, class Grid>
            GT_FUNCTION_DEVICE std::enable_if_t<execute::is_parallel<ExecutionType>::value, int_t> compute_kblock(
                ExecutionType, Grid const &grid) {
                return blockIdx.z * ExecutionType::block_size + grid.k_min();
            };

            template <class ExecutionType, class LoopIntervals, class LocalDomain, class Grid>
            struct kernel_f {
                GT_STATIC_ASSERT(std::is_trivially_copyable<LocalDomain>::value, GT_INTERNAL_ERROR);
                GT_STATIC_ASSERT(std::is_trivially_copyable<Grid>::value, GT_INTERNAL_ERROR);

                LocalDomain m_local_domain;
                Grid m_grid;
                LoopIntervals m_loop_intervals;

                template <class Validator>
                GT_FUNCTION_DEVICE void operator()(int_t iblock, int_t jblock, Validator &&validator) const {
                    iterate_domain<LocalDomain> it_domain(
                        m_local_domain, iblock, jblock, compute_kblock(ExecutionType(), m_grid));
                    // execute the k interval functors
                    run_functors_on_interval<ExecutionType>(it_domain, m_loop_intervals, validator);
                }
            };
        } // namespace fused_mss_loop_cuda_impl_

        template <class ExecutionType, class LocalDomain, class Grid, class LoopIntervals>
        fused_mss_loop_cuda_impl_::kernel_f<ExecutionType, LoopIntervals, LocalDomain, Grid> make_kernel(
            LocalDomain const &local_domain, Grid const &grid, LoopIntervals const &loop_intervals) {
            return {local_domain, grid, loop_intervals};
        }
    } // namespace cuda
} // namespace gridtools

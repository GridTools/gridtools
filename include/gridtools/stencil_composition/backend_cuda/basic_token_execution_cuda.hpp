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

#include "../../common/array.hpp"
#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/gt_math.hpp"
#include "../../common/host_device.hpp"
#include "../../common/pair.hpp"
#include "../../meta.hpp"
#include "../execution_types.hpp"
#include "../interval.hpp"
#include "../iteration_policy.hpp"
#include "../level.hpp"
#include "../loop_interval.hpp"
#include "../run_functor_arguments.hpp"
#include "run_esf_functor_cuda.hpp"

namespace gridtools {
    namespace cuda {
        /**
         * get_k_interval specialization for parallel execution policy. The full k-axis is split into equally-sized
         * block sub-intervals and assigned to the blocks in z-direction. Each block iterates over all computation
         * intervals and calculates the subinterval which intersects with the block sub-interval.
         *
         * Example with with an axis with four intervals, that is distributed among 2 blocks:
         *
         * Computation intervals   block sub-intervals
         *                           with two blocks
         *
         *                                       B1   B2
         *    0 ---------           0 --------- ---         Block B1 calculates the complete intervals I1 and I2, and
         * parts |                     |    1 :          of I3. It does not calculate anything for I4. |  I1 B1  | : |
         * |      :          1. iteration: get_k_interval(...) = [0, 4] |                     |      :          2.
         * iteration: get_k_interval(...) = [5, 7] 5    ---                    |     ---         3. iteration:
         * get_k_interval(...) = [8, 9] |                     |    2 :          4. iteration: get_k_interval(...) = [18,
         * 9] (= no calculation) | I2                  |      : 8    ---                    |     --- | |    3 : | I3 10
         * ---    ---  ---    Block B2 calculates parts of the interval I3, and the complete |                     | 3 :
         * interval I4. It does not calculate anything for I1 and I2. |                 B2  |           : | | :     1.
         * iteration: get_k_interval(...) = [10, 4] (= no calculation) |                     |           :     2.
         * iteration: get_k_interval(...) = [10, 7] (= no calculation) |                     |           :     3.
         * iteration: get_k_interval(...) = [10, 17] |                     |           :     4. iteration:
         * get_k_interval(...) = [18, 20] |                     |           : 18    ---                    | --- | I4 |
         * 4 : 20 ---------          20 ---------      ---
         */
        template <class FromLevel, class ToLevel, uint_t BlockSize, class Grid>
        GT_FUNCTION_DEVICE pair<int, int> get_k_interval(execute::parallel_block<BlockSize>, Grid const &grid) {
            return {math::max<int>(blockIdx.z * BlockSize, grid.template value_at<FromLevel>()),
                math::min<int>((blockIdx.z + 1) * BlockSize - 1, grid.template value_at<ToLevel>())};
        }

        template <class FromLevel, class ToLevel, class ExecutionEngine, class Grid>
        GT_FUNCTION_DEVICE pair<int, int> get_k_interval(ExecutionEngine, Grid const &grid) {
            return {grid.template value_at<FromLevel>(), grid.template value_at<ToLevel>()};
        }

        namespace _impl {

            /**
               @brief basic token of execution responsible of handling the discretization over the vertical dimension.
               This may be done with a loop over k or by partitioning the k axis and executing in parallel, depending on
               the execution_policy defined in the multi-stage stencil.
            */
            /**
               @brief   Execution kernel containing the loop over k levels
            */
            template <class ExecutionType, class LoopIntervals, class ItDomain, class Grid>
            struct run_f_on_interval_with_k_caches {
                using first_t = meta::first<LoopIntervals>;
                using last_t = meta::last<LoopIntervals>;

                ItDomain &m_domain;
                bool m_in_domain;
                Grid const &m_grid;

                template <class IterationPolicy,
                    class Stages,
                    bool IsFirst,
                    bool IsLast,
                    std::enable_if_t<meta::length<Stages>::value != 0, int> = 0>
                GT_FUNCTION_DEVICE void k_loop(int_t first, int_t last) const {
                    for (int_t cur = first; IterationPolicy::condition(cur, last);
                         IterationPolicy::increment(cur), m_domain.increment_k(execute::step<ExecutionType>)) {
                        if (m_in_domain)
                            m_domain.template fill_caches<ExecutionType>(IsFirst && cur == first);
                        run_esf_functor_cuda<Stages>(m_domain);
                        if (m_in_domain)
                            m_domain.template flush_caches<ExecutionType>(IsLast && cur == last);
                        m_domain.template slide_caches<ExecutionType>();
                    }
                }

                template <class IterationPolicy,
                    class Stages,
                    bool IsFirst,
                    bool IsLast,
                    std::enable_if_t<meta::length<Stages>::value == 0, int> = 0>
                GT_FUNCTION_DEVICE void k_loop(int_t first, int_t last) const {
                    for (int_t cur = first; IterationPolicy::condition(cur, last);
                         IterationPolicy::increment(cur), m_domain.increment_k(execute::step<ExecutionType>)) {
                        if (m_in_domain) {
                            m_domain.template fill_caches<ExecutionType>(IsFirst && cur == first);
                            m_domain.template flush_caches<ExecutionType>(IsLast && cur == last);
                        }
                        m_domain.template slide_caches<ExecutionType>();
                    }
                }

                template <class LoopInterval>
                GT_FUNCTION_DEVICE void operator()() const {
                    GT_STATIC_ASSERT(is_loop_interval<LoopInterval>::value, GT_INTERNAL_ERROR);
                    using from_t = meta::first<LoopInterval>;
                    using to_t = meta::second<LoopInterval>;
                    using stage_groups_t = meta::at_c<LoopInterval, 2>;
                    using iteration_policy_t = iteration_policy<from_t, to_t, ExecutionType>;
                    constexpr auto is_first = std::is_same<LoopInterval, first_t>::value;
                    constexpr auto is_last = std::is_same<LoopInterval, last_t>::value;
                    k_loop<iteration_policy_t, stage_groups_t, is_first, is_last>(
                        m_grid.template value_at<from_t>(), m_grid.template value_at<to_t>());
                }
            };

            template <class ExecutionType, class ItDomain, class Grid>
            struct run_f_on_interval {
                ItDomain &m_domain;
                Grid const &m_grid;

                template <class IterationPolicy,
                    class Stages,
                    std::enable_if_t<meta::length<Stages>::value != 0, int> = 0>
                GT_FUNCTION_DEVICE void k_loop(int_t first, int_t last) const {
                    for (int_t cur = first; IterationPolicy::condition(cur, last);
                         IterationPolicy::increment(cur), m_domain.increment_k(execute::step<ExecutionType>))
                        run_esf_functor_cuda<Stages>(m_domain);
                }

                template <class IterationPolicy,
                    class Stages,
                    std::enable_if_t<meta::length<Stages>::value == 0, int> = 0>
                GT_FUNCTION_DEVICE void k_loop(int_t first, int_t last) const {
                    for (int_t cur = first; IterationPolicy::condition(cur, last);
                         IterationPolicy::increment(cur), m_domain.increment_k(execute::step<ExecutionType>)) {
                    }
                }

                template <class LoopInterval>
                GT_FUNCTION_DEVICE void operator()() const {
                    GT_STATIC_ASSERT(is_loop_interval<LoopInterval>::value, GT_INTERNAL_ERROR);
                    using from_t = meta::first<LoopInterval>;
                    using to_t = meta::second<LoopInterval>;
                    using stage_groups_t = meta::at_c<LoopInterval, 2>;
                    using iteration_policy_t = iteration_policy<from_t, to_t, ExecutionType>;
                    const auto k_interval = get_k_interval<from_t, to_t>(ExecutionType{}, m_grid);
                    k_loop<iteration_policy_t, stage_groups_t>(k_interval.first, k_interval.second);
                }
            };
        } // namespace _impl

        template <class ExecutionType, class LoopIntervals, class MaxExtent, class ItDomain, class Grid>
        GT_FUNCTION_DEVICE std::enable_if_t<ItDomain::has_k_caches> run_functors_on_interval(
            ItDomain &it_domain, Grid const &grid) {
            bool in_domain = it_domain.template is_thread_in_domain<MaxExtent>();
            device::for_each_type<LoopIntervals>(
                _impl::run_f_on_interval_with_k_caches<ExecutionType, LoopIntervals, ItDomain, Grid>{
                    it_domain, in_domain, grid});
        }

        template <class ExecutionType, class LoopIntervals, class MaxExtent, class ItDomain, class Grid>
        GT_FUNCTION_DEVICE std::enable_if_t<!ItDomain::has_k_caches> run_functors_on_interval(
            ItDomain &it_domain, Grid const &grid) {
            device::for_each_type<LoopIntervals>(
                _impl::run_f_on_interval<ExecutionType, ItDomain, Grid>{it_domain, grid});
        }
    } // namespace cuda
} // namespace gridtools

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
#include "../../common/host_device.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../execution_types.hpp"
#include "k_cache.hpp"

namespace gridtools {
    namespace cuda {
        namespace _impl {
            template <class ExecutionType, class Grid>
            auto start(ExecutionType, Grid const &grid) {
                return grid.k_min();
            };

            template <class Grid>
            auto start(execute::backward, Grid const &grid) {
                return grid.k_max();
            };
        } // namespace _impl

        template <class ExecutionType, class LoopIntervals, class KCaches>
        struct cached_k_loop_f {
            LoopIntervals m_loop_intervals;
            int_t m_start;

            template <class Ptr, class Strides, class Validator>
            GT_FUNCTION_DEVICE void operator()(Ptr ptr, Strides const &strides, Validator validator) const {
                sid::shift(ptr, sid::get_stride<dim::k>(strides), m_start);
                KCaches k_caches;
                auto mixed_ptr = hymap::device::merge(k_caches.ptr(), wstd::move(ptr));
                tuple_util::device::for_each(
                    [&](auto const &loop_interval) {
                        for (int_t i = 0; i < loop_interval.count(); ++i) {
                            loop_interval(mixed_ptr, strides, validator);
                            k_caches.slide(execute::step<ExecutionType>);
                            sid::shift(
                                mixed_ptr.secondary(), sid::get_stride<dim::k>(strides), execute::step<ExecutionType>);
                        }
                    },
                    m_loop_intervals);
            }
        };

        template <class ExecutionType, class LoopIntervals>
        struct k_loop_f {
            LoopIntervals m_loop_intervals;
            int_t m_start;

            template <class Ptr, class Strides, class Validator>
            GT_FUNCTION_DEVICE void operator()(Ptr ptr, Strides const &strides, Validator validator) const {
                sid::shift(ptr, sid::get_stride<dim::k>(strides), m_start);
                tuple_util::device::for_each(
                    [&](auto const &loop_interval) {
                        for (int_t i = 0; i < loop_interval.count(); ++i) {
                            loop_interval(ptr, strides, validator);
                            sid::shift(ptr, sid::get_stride<dim::k>(strides), execute::step<ExecutionType>);
                        }
                    },
                    m_loop_intervals);
            }
        };

        template <int_t BlockSize, class LoopIntervals, class Start>
        struct parallel_k_loop_f {
            LoopIntervals m_loop_intervals;
            Start m_start;

            template <class Ptr, class Strides, class Validator>
            GT_FUNCTION_DEVICE void operator()(Ptr ptr, Strides const &strides, Validator validator) const {
                sid::shift(ptr, sid::get_stride<dim::k>(strides), m_start + (int_t)blockIdx.z * BlockSize);
                int cur = -(int_t)blockIdx.z * BlockSize;
                tuple_util::device::for_each(
                    [&](auto const &loop_interval) {
                        if (cur >= BlockSize)
                            return;
                        auto count = loop_interval.count();
                        int_t lim = math::min(cur + count, BlockSize) - math::max(cur, 0);
                        cur += count;
#pragma unroll BlockSize
                        for (int_t i = 0; i < BlockSize; ++i) {
                            if (i >= lim)
                                break;
                            loop_interval(ptr, strides, validator);
                            sid::shift(ptr, sid::get_stride<dim::k>(strides), integral_constant<int_t, 1>());
                        }
                    },
                    m_loop_intervals);
            }
        };

        template <class Mss,
            class Grid,
            class LoopIntervals,
            std::enable_if_t<meta::any_of<is_k_cache, typename Mss::cache_sequence_t>::value, int> = 0>
        auto make_k_loop(Grid const &grid, LoopIntervals loop_intervals) {
            using execution_t = typename Mss::execution_engine_t;
            return cached_k_loop_f<execution_t, LoopIntervals, k_caches_type<Mss>>{
                std::move(loop_intervals), _impl::start(execution_t(), grid)};
        }

        template <class Mss,
            class Grid,
            class LoopIntervals,
            std::enable_if_t<!meta::any_of<is_k_cache, typename Mss::cache_sequence_t>::value &&
                                 !execute::is_parallel<typename Mss::execution_engine_t>::value,
                int> = 0>
        auto make_k_loop(Grid const &grid, LoopIntervals loop_intervals) {
            using execution_t = typename Mss::execution_engine_t;
            return k_loop_f<typename Mss::execution_engine_t, LoopIntervals>{
                std::move(loop_intervals), _impl::start(execution_t(), grid)};
        }

        template <class Mss,
            class Grid,
            class LoopIntervals,
            std::enable_if_t<!meta::any_of<is_k_cache, typename Mss::cache_sequence_t>::value &&
                                 execute::is_parallel<typename Mss::execution_engine_t>::value,
                int> = 0>
        auto make_k_loop(Grid const &grid, LoopIntervals loop_intervals) {
            return parallel_k_loop_f<Mss::execution_engine_t::block_size, LoopIntervals, decltype(grid.k_min())>{
                std::move(loop_intervals), grid.k_min()};
        }

    } // namespace cuda
} // namespace gridtools

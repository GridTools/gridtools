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
            template <class ExecutionType>
            GT_FUNCTION_DEVICE auto make_count_modifier(ExecutionType) {
                return [](auto x) { return x; };
            }

            template <uint_t BlockSize>
            GT_FUNCTION_DEVICE auto make_count_modifier(execute::parallel_block<BlockSize>) {
                return [cur = -(int_t)blockIdx.z * (int_t)BlockSize](int_t x) mutable {
                    if (cur >= (int_t)BlockSize)
                        return 0;
                    int_t res = math::min(cur + x, (int_t)BlockSize) - math::max(cur, 0);
                    cur += x;
                    return res;
                };
            }

            template <class ExecutionType>
            GT_FUNCTION_DEVICE int_t start_offset(ExecutionType) {
                return 0;
            };

            template <uint_t BlockSize>
            GT_FUNCTION_DEVICE int_t start_offset(execute::parallel_block<BlockSize>) {
                return blockIdx.z * BlockSize;
            };

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
                        auto count = loop_interval.count();
                        for (int_t i = 0; i < count; ++i) {
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
                sid::shift(ptr, sid::get_stride<dim::k>(strides), m_start + _impl::start_offset(ExecutionType()));
                auto count_modifier = _impl::make_count_modifier(ExecutionType{});
                tuple_util::device::for_each(
                    [&](auto const &loop_interval) {
                        auto count = count_modifier(loop_interval.count());
                        for (int_t i = 0; i < count; ++i) {
                            loop_interval(ptr, strides, validator);
                            sid::shift(ptr, sid::get_stride<dim::k>(strides), execute::step<ExecutionType>);
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
            std::enable_if_t<!meta::any_of<is_k_cache, typename Mss::cache_sequence_t>::value, int> = 0>
        auto make_k_loop(Grid const &grid, LoopIntervals loop_intervals) {
            using execution_t = typename Mss::execution_engine_t;
            return k_loop_f<typename Mss::execution_engine_t, LoopIntervals>{
                std::move(loop_intervals), _impl::start(execution_t(), grid)};
        }
    } // namespace cuda
} // namespace gridtools

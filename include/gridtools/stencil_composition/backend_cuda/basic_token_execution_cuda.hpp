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
            template <class I>
            using is_first_f = bool_constant<I::value == 0>;

            template <class Length, class I>
            using is_last_f = bool_constant<I::value + 1 == Length::value>;

            template <class Args, class Fun>
            GT_FUNCTION_DEVICE void for_each_with_first_last(Fun const &fun, Args const &args) {
                using indices_t = meta::make_indices_for<Args, tuple>;
                tuple_util::device::for_each(fun,
                    args,
                    meta::transform<is_first_f, indices_t>{},
                    meta::transform<meta::curry<is_last_f, meta::length<Args>>::template apply, indices_t>{});
            }

            template <class Fun>
            GT_FUNCTION_DEVICE void loop_with_first_last(Fun const &fun, int_t size) {
                switch (size) {
                case 0:
                    break;
                case 1:
                    fun(std::true_type{}, std::true_type{});
                    break;
                case 2:
                    fun(std::true_type{}, std::false_type{});
                    fun(std::false_type{}, std::true_type{});
                    break;
                default:
                    fun(std::true_type{}, std::false_type{});
                    for (int_t i = 2; i < size; ++i)
                        fun(std::false_type{}, std::false_type{});
                    fun(std::false_type{}, std::true_type{});
                }
            }

            template <class Fun, class I, class Int, Int Size>
            GT_FUNCTION_DEVICE void loop_with_first_last(Fun const &fun, std::integral_constant<Int, Size>) {
                for_each_with_first_last([&fun](auto, auto is_first, auto is_last) { fun(is_first, is_last); },
                    meta::iseq_to_list<std::make_integer_sequence<Int, Size>, tuple>{});
            }

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

        template <class ExecutionType, class LoopIntervals, class KCachesMaker>
        struct cached_k_loop_f {
            LoopIntervals m_loop_intervals;
            KCachesMaker m_k_caches_maker;
            int_t m_start;

            template <class Ptr, class Strides, class Validator>
            GT_FUNCTION_DEVICE void operator()(Ptr ptr, Strides const &strides, Validator validator) const {
                sid::shift(ptr, sid::get_stride<dim::k>(strides), m_start);
                auto k_caches = m_k_caches_maker();
                _impl::for_each_with_first_last(
                    [&](auto const &loop_interval, auto is_first_interval, auto is_last_interval) {
                        _impl::loop_with_first_last(
                            [&](auto is_first_level, auto is_last_level) {
                                loop_interval(k_caches.mixin_ptrs(ptr), strides, validator);
                                k_caches.slide(execute::step<ExecutionType>);
                                sid::shift(ptr, sid::get_stride<dim::k>(strides), execute::step<ExecutionType>);
                            },
                            loop_interval.count());
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
            class Composite,
            std::enable_if_t<meta::any_of<is_k_cache, typename Mss::cache_sequence_t>::value, int> = 0>
        auto make_k_loop(Mss, Grid const &grid, LoopIntervals loop_intervals, Composite &&composite) {
            using execution_t = typename Mss::execution_engine_t;
            auto caches_maker = make_k_caches_maker(Mss(), std::forward<Composite>(composite));
            return cached_k_loop_f<execution_t, LoopIntervals, decltype(caches_maker)>{
                std::move(loop_intervals), std::move(caches_maker), _impl::start(execution_t(), grid)};
        }

        template <class Mss,
            class Grid,
            class LoopIntervals,
            class Composite,
            std::enable_if_t<!meta::any_of<is_k_cache, typename Mss::cache_sequence_t>::value, int> = 0>
        auto make_k_loop(Mss, Grid const &grid, LoopIntervals loop_intervals, Composite &&) {
            using execution_t = typename Mss::execution_engine_t;
            return k_loop_f<typename Mss::execution_engine_t, LoopIntervals>{
                std::move(loop_intervals), _impl::start(execution_t(), grid)};
        }
    } // namespace cuda
} // namespace gridtools

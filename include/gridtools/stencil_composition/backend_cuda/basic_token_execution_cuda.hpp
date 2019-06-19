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

            template <uint_t BlockSize>
            GT_FUNCTION_DEVICE auto strip_from(execute::parallel_block<BlockSize>, int_t src) {
                return math::max<int_t>(blockIdx.z * BlockSize, src);
            }

            template <uint_t BlockSize>
            GT_FUNCTION_DEVICE auto strip_to(execute::parallel_block<BlockSize>, int_t src) {
                return math::min<int_t>((blockIdx.z + 1) * BlockSize - 1, src);
            }

            template <class ExecutionType>
            struct modify_count_f {
                template <class Count>
                GT_FUNCTION_DEVICE Count operator()(Count count) const {
                    return count;
                }
            };

            template <uint_t BlockSize>
            struct modify_count_f<execute::parallel_block<BlockSize>> {
                mutable int_t m_cur = -blockIdx.z * BlockSize;

                template <class Count>
                GT_FUNCTION_DEVICE int_t operator()(Count count) const {
                    if (m_cur >= BlockSize)
                        return 0;
                    int_t res = math::min<int_t>(m_cur + count, BlockSize) - math::max(m_cur, 0);
                    m_cur += count;
                    return res;
                }
            };
        } // namespace _impl

        template <class ExecutionType, class MaxExtent, class ItDomain, class Grid, class LoopIntervals>
        GT_FUNCTION_DEVICE std::enable_if_t<ItDomain::has_k_caches> run_functors_on_interval(
            ItDomain &it_domain, Grid const &, LoopIntervals const &loop_intervals) {
            bool in_domain = it_domain.template is_thread_in_domain<MaxExtent>();
            _impl::for_each_with_first_last(
                [&](auto const &loop_interval, auto is_first_interval, auto is_last_interval) {
                    _impl::loop_with_first_last(
                        [&](auto is_first_level, auto is_last_level) {
                            if (in_domain)
                                it_domain.fill_caches(ExecutionType(),
                                    bool_constant<decltype(is_first_interval)::value &&decltype(
                                        is_first_level)::value>());
                            loop_interval(it_domain.ptr(), it_domain.strides(), [&](auto extent) {
                                return it_domain.template is_thread_in_domain<decltype(extent)>();
                            });
                            if (in_domain)
                                it_domain.flush_caches(ExecutionType(),
                                    bool_constant<decltype(is_last_interval)::value &&decltype(
                                        is_last_level)::value>());
                            it_domain.increment_k(ExecutionType());
                        },
                        loop_interval.count());
                },
                loop_intervals);
        }

        template <class ExecutionType, class MaxExtent, class ItDomain, class Grid, class LoopIntervals>
        GT_FUNCTION_DEVICE std::enable_if_t<!ItDomain::has_k_caches> run_functors_on_interval(
            ItDomain &it_domain, Grid const &grid, LoopIntervals const &loop_intervals) {
            _impl::modify_count_f<ExecutionType> modify_count;
            tuple_util::device::for_each(
                [&](auto const &loop_interval) {
                    auto count = modify_count(loop_interval.count());
                    for (int_t i = 0; i < count; ++i) {
                        loop_interval(it_domain.ptr(), it_domain.strides(), [&](auto extent) {
                            return it_domain.template is_thread_in_domain<decltype(extent)>();
                        });
                        it_domain.increment_k(ExecutionType());
                    }
                },
                loop_intervals);
        }
    } // namespace cuda
} // namespace gridtools

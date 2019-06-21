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
        } // namespace _impl

        template <class ExecutionType, class ItDomain, class LoopIntervals, class Validator>
        GT_FUNCTION_DEVICE std::enable_if_t<ItDomain::has_k_caches> run_functors_on_interval(
            ItDomain &it_domain, LoopIntervals const &loop_intervals, Validator validator) {
            _impl::for_each_with_first_last(
                [&](auto const &loop_interval, auto is_first_interval, auto is_last_interval) {
                    _impl::loop_with_first_last(
                        [&](auto is_first_level, auto is_last_level) {
                            if (validator())
                                it_domain.fill_caches(ExecutionType(),
                                    bool_constant<decltype(is_first_interval)::value &&decltype(
                                        is_first_level)::value>());
                            loop_interval(it_domain.ptr(), it_domain.strides(), validator);
                            if (validator())
                                it_domain.flush_caches(ExecutionType(),
                                    bool_constant<decltype(is_last_interval)::value &&decltype(
                                        is_last_level)::value>());
                            it_domain.increment_k(ExecutionType());
                        },
                        loop_interval.count());
                },
                loop_intervals);
        }

        template <class ExecutionType, class ItDomain, class LoopIntervals, class Validator>
        GT_FUNCTION_DEVICE std::enable_if_t<!ItDomain::has_k_caches> run_functors_on_interval(
            ItDomain &it_domain, LoopIntervals const &loop_intervals, Validator validator) {
            auto count_modifier = _impl::make_count_modifier(ExecutionType{});
            tuple_util::device::for_each(
                [&](auto const &loop_interval) {
                    auto count = count_modifier(loop_interval.count());
                    for (int_t i = 0; i < count; ++i) {
                        loop_interval(it_domain.ptr(), it_domain.strides(), validator);
                        it_domain.increment_k(ExecutionType());
                    }
                },
                loop_intervals);
        }
    } // namespace cuda
} // namespace gridtools

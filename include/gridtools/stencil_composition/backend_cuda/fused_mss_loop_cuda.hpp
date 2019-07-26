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

#include <utility>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/utility.hpp"
#include "../../common/gt_math.hpp"
#include "../../common/host_device.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../dim.hpp"
#include "../execution_types.hpp"
#include "../sid/blocked_dim.hpp"
#include "../sid/concept.hpp"
#include "k_cache.hpp"

namespace gridtools {
    namespace cuda {
        namespace fused_mss_loop_cuda_impl_ {
            template <class ExecutionType, class Grid>
            integral_constant<int_t, 0> start(ExecutionType, Grid &&) {
                return {};
            };

            template <class Grid>
            auto start(execute::backward, Grid const &grid) {
                return grid.k_size();
            };

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
                                sid::shift(mixed_ptr.secondary(),
                                    sid::get_stride<dim::k>(strides),
                                    execute::step<ExecutionType>);
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

            template <int_t BlockSize, class LoopIntervals>
            struct parallel_k_loop_f {
                LoopIntervals m_loop_intervals;

                template <class Ptr, class Strides, class Validator>
                GT_FUNCTION_DEVICE void operator()(Ptr ptr, Strides const &strides, Validator validator) const {
                    sid::shift(ptr, sid::get_stride<dim::k>(strides), (int_t)blockIdx.z * BlockSize);
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
                class DataStores,
                class Grid,
                class LoopIntervals,
                std::enable_if_t<meta::any_of<is_k_cache, typename Mss::cache_sequence_t>::value, int> = 0>
            auto make_k_loop(Grid const &grid, LoopIntervals loop_intervals) {
                using execution_t = typename Mss::execution_engine_t;
                return cached_k_loop_f<execution_t, LoopIntervals, k_caches_type<Mss, DataStores>>{
                    std::move(loop_intervals), start(execution_t(), grid)};
            }

            template <class Mss,
                class /*DataStores*/,
                class Grid,
                class LoopIntervals,
                std::enable_if_t<!meta::any_of<is_k_cache, typename Mss::cache_sequence_t>::value &&
                                     !execute::is_parallel<typename Mss::execution_engine_t>::value,
                    int> = 0>
            auto make_k_loop(Grid const &grid, LoopIntervals loop_intervals) {
                using execution_t = typename Mss::execution_engine_t;
                return k_loop_f<typename Mss::execution_engine_t, LoopIntervals>{
                    std::move(loop_intervals), start(execution_t(), grid)};
            }

            template <class Mss,
                class /*DataStores*/,
                class Grid,
                class LoopIntervals,
                std::enable_if_t<!meta::any_of<is_k_cache, typename Mss::cache_sequence_t>::value &&
                                     execute::is_parallel<typename Mss::execution_engine_t>::value,
                    int> = 0>
            auto make_k_loop(Grid const &grid, LoopIntervals loop_intervals) {
                return parallel_k_loop_f<Mss::execution_engine_t::block_size, LoopIntervals>{std::move(loop_intervals)};
            }

            template <class Sid, class KLoop>
            struct kernel_f {
                sid::ptr_holder_type<Sid> m_ptr_holder;
                sid::strides_type<Sid> m_strides;
                KLoop k_loop;

                template <class Validator>
                GT_FUNCTION_DEVICE void operator()(int_t i_block, int_t j_block, Validator validator) const {
                    sid::ptr_diff_type<Sid> offset = {};
                    sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::i>>(m_strides), blockIdx.x);
                    sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::j>>(m_strides), blockIdx.y);
                    sid::shift(offset, sid::get_stride<dim::i>(m_strides), i_block);
                    sid::shift(offset, sid::get_stride<dim::j>(m_strides), j_block);
                    k_loop(m_ptr_holder() + offset, m_strides, wstd::move(validator));
                }
            };

            template <class Composite, class KLoop>
            kernel_f<Composite, KLoop> make_kernel_fun_raw(Composite &composite, KLoop k_loop) {
                return {sid::get_origin(composite), sid::get_strides(composite), std::move(k_loop)};
            }

            template <class Mss, class DataStores, class Grid, class Composite, class LoopIntervals>
            auto make_kernel_fun(Grid const &grid, Composite &composite, LoopIntervals loop_intervals) {
                return make_kernel_fun_raw(composite, make_k_loop<Mss, DataStores>(grid, std::move(loop_intervals)));
            }

        } // namespace fused_mss_loop_cuda_impl_

        using fused_mss_loop_cuda_impl_::make_kernel_fun;

    } // namespace cuda
} // namespace gridtools

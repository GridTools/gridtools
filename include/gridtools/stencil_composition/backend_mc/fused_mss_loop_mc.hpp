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
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../meta.hpp"
#include "../caches/cache_metafunctions.hpp"
#include "../dim.hpp"
#include "../grid.hpp"
#include "../loop_interval.hpp"
#include "../mss_components.hpp"
#include "../mss_components_metafunctions.hpp"
#include "../run_functor_arguments.hpp"
#include "../sid/blocked_dim.hpp"
#include "../sid/loop.hpp"
#include "execinfo_mc.hpp"
#include "tmp_storage_sid.hpp"

namespace gridtools {
    namespace mc {
        template <class Stage, class Ptr, class Strides>
        GT_FORCE_INLINE void i_loop(int_t size, Stage stage, Ptr &ptr, Strides const &strides) {
#ifdef NDEBUG
#pragma ivdep
#ifndef __INTEL_COMPILER
#pragma omp simd
#endif
#endif
            for (int_t i = 0; i < size; ++i) {
                using namespace literals;
                stage(ptr, strides);
                sid::shift(ptr, sid::get_stride<dim::i>(strides), 1_c);
            }
            sid::shift(ptr, sid::get_stride<dim::i>(strides), -size);
        }

        template <class Funs>
        void exec_serial(execinfo_mc info, Funs &&funs) {
            int_t i_blocks = info.i_blocks();
            int_t j_blocks = info.j_blocks();
#pragma omp parallel for collapse(2)
            for (int_t j = 0; j < j_blocks; ++j) {
                for (int_t i = 0; i < i_blocks; ++i) {
                    tuple_util::for_each([block = info.block(i, j)](auto &&fun) { fun(block); }, funs);
                }
            }
        }

        template <class KStep, class Ptr, class Strides>
        struct k_i_loops_f {
            int_t m_i_size;
            Ptr &m_ptr;
            Strides const &m_strides;

            template <class Interval>
            GT_FORCE_INLINE void operator()(Interval interval) const {
                for (int_t k = 0; k < interval.count(); ++k) {
                    i_loop(m_i_size, interval, m_ptr, m_strides);
                    sid::shift(m_ptr, sid::get_stride<dim::k>(m_strides), KStep());
                }
            }
        };

        template <class KStep, class Ptr, class Strides>
        GT_FORCE_INLINE k_i_loops_f<KStep, Ptr, Strides> make_k_i_loops(
            int_t i_size, Ptr &ptr, Strides const &strides) {
            return {i_size, ptr, strides};
        }

        integral_constant<int_t, 0> k_start(int_t, integral_constant<int_t, 1>) { return {}; }

        int_t k_start(int_t total_length, integral_constant<int_t, -1>) { return total_length - 1; }

        template <class Extent, class KStep, class Composite, class LoopIntervals>
        auto make_mss_serial_loop(int_t k_total_length, KStep, Composite composite, LoopIntervals loop_intervals) {
            auto strides = sid::get_strides(composite);
            sid::ptr_diff_type<Composite> offset{};
            sid::shift(offset, sid::get_stride<dim::i>(strides), typename Extent::iminus());
            sid::shift(offset, sid::get_stride<dim::j>(strides), typename Extent::jminus());
            sid::shift(offset, sid::get_stride<dim::k>(strides), k_start(k_total_length, KStep()));

            return [origin = sid::get_origin(composite) + offset,
                       strides = std::move(strides),
                       k_shift_back = -k_total_length * KStep(),
                       loop_intervals](execinfo_block_kserial_mc const &info) {
                sid::ptr_diff_type<Composite> offset{};
                sid::shift(offset, sid::get_stride<thread_dim_mc>(strides), omp_get_thread_num());
                sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::i>>(strides), info.i_block);
                sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::j>>(strides), info.j_block);
                auto ptr = origin() + offset;

                int_t j_size = info.j_block_size + Extent::jplus::value - Extent::jminus::value;
                int_t i_size = info.i_block_size + Extent::iplus::value - Extent::iminus::value;

                auto k_i_loops = make_k_i_loops<KStep>(i_size, ptr, strides);
                for (int_t j = 0; j < j_size; ++j) {
                    using namespace literals;
                    tuple_util::for_each(k_i_loops, loop_intervals);
                    sid::shift(ptr, sid::get_stride<dim::k>(strides), k_shift_back);
                    sid::shift(ptr, sid::get_stride<dim::j>(strides), 1_c);
                }
            };
        }

        template <class Funs>
        void exec_parallel(execinfo_mc info, int_t k_count, Funs &&funs) {
            int_t i_blocks = info.i_blocks();
            int_t j_blocks = info.j_blocks();
#pragma omp parallel for collapse(3)
            for (int_t j = 0; j < j_blocks; ++j) {
                for (int_t k = 0; k < k_count; ++k) {
                    for (int_t i = 0; i < i_blocks; ++i) {
                        tuple_util::for_each([block = info.block(i, j, k)](auto &&fun) { fun(block); }, funs);
                    }
                }
            }
        }

        template <class Extent, class Composite, class LoopIntervals>
        auto make_mss_parallel_loop(Composite composite, LoopIntervals loop_intervals) {
            auto strides = sid::get_strides(composite);
            sid::ptr_diff_type<Composite> offset{};
            sid::shift(offset, sid::get_stride<dim::i>(strides), typename Extent::iminus());
            sid::shift(offset, sid::get_stride<dim::j>(strides), typename Extent::jminus());
            return [origin = sid::get_origin(composite) + offset, strides = std::move(strides), loop_intervals](
                       execinfo_block_kparallel_mc const &info) {
                sid::ptr_diff_type<Composite> offset{};
                sid::shift(offset, sid::get_stride<thread_dim_mc>(strides), omp_get_thread_num());
                sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::i>>(strides), info.i_block);
                sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::j>>(strides), info.j_block);
                sid::shift(offset, sid::get_stride<dim::k>(strides), info.k);
                auto ptr = origin() + offset;

                int_t j_count = info.j_block_size + Extent::jplus::value - Extent::jminus::value;
                int_t i_size = info.i_block_size + Extent::iplus::value - Extent::iminus::value;

                for (int_t j = 0; j < j_count; ++j) {
                    using namespace literals;
                    int_t cur = 0;
                    tuple_util::for_each(
                        [&ptr, &strides, &cur, k = info.k, i_size](auto loop_interval) {
                            if (k >= cur && k < cur + loop_interval.count())
                                i_loop(i_size, loop_interval, ptr, strides);
                            cur += loop_interval.count();
                        },
                        loop_intervals);
                    sid::shift(ptr, sid::get_stride<dim::j>(strides), 1_c);
                }
            };
        }
    } // namespace mc
} // namespace gridtools

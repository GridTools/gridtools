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
#include <utility>

#include "../../../common/defs.hpp"
#include "../../../common/generic_metafunctions/for_each.hpp"
#include "../../../common/host_device.hpp"
#include "../../../common/integral_constant.hpp"
#include "../../../common/tuple_util.hpp"
#include "../../../meta.hpp"
#include "../../dim.hpp"
#include "../../sid/concept.hpp"
#include "../../sid/loop.hpp"
#include "../dim.hpp"

namespace gridtools {
    namespace x86 {

        template <class KStep, class Intervals>
        struct k_loop_f {
            Intervals m_intervals;
            int_t m_shift_back;

            template <class Ptr, class Strides>
            GT_FORCE_INLINE void operator()(Ptr &ptr, Strides const &strides) const {
                tuple_util::for_each(
                    [&ptr, &strides](auto interval) {
                        for (int k = 0; k < interval.count(); ++k) {
                            interval(ptr, strides);
                            sid::shift(ptr, sid::get_stride<dim::k>(strides), KStep());
                        }
                    },
                    m_intervals);
                sid::shift(ptr, sid::get_stride<dim::k>(strides), m_shift_back);
            }
        };

        integral_constant<int_t, 0> k_start(int_t, integral_constant<int_t, 1>) { return {}; }

        int_t k_start(int_t total_length, integral_constant<int_t, -1>) { return total_length - 1; }

        template <class Extent, class KStep, class Composite, class LoopIntervals>
        auto make_mss_loop(int_t k_total_length, KStep, Composite composite, LoopIntervals loop_intervals) {
            auto strides = sid::get_strides(composite);
            sid::ptr_diff_type<Composite> offset{};
            sid::shift(offset, sid::get_stride<dim::i>(strides), typename Extent::iminus());
            sid::shift(offset, sid::get_stride<dim::j>(strides), typename Extent::jminus());
            sid::shift(offset, sid::get_stride<dim::k>(strides), k_start(k_total_length, KStep()));

            return [origin = sid::get_origin(composite) + offset,
                       strides = std::move(strides),
                       k_loop = k_loop_f<KStep, LoopIntervals>{std::move(loop_intervals), -k_total_length * KStep()}](
                       int_t i_block, int_t j_block, int_t i_size, int_t j_size) {
                sid::ptr_diff_type<Composite> offset{};
                sid::shift(offset, sid::get_stride<dim::thread>(strides), omp_get_thread_num());
                sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::i>>(strides), i_block);
                sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::j>>(strides), j_block);
                auto ptr = origin() + offset;

                auto i_loop = sid::make_loop<dim::i>(i_size + Extent::iplus::value - Extent::iminus::value);
                auto j_loop = sid::make_loop<dim::j>(j_size + Extent::jplus::value - Extent::jminus::value);

                i_loop(j_loop(k_loop))(ptr, strides);
            };
        }

    } // namespace x86
} // namespace gridtools

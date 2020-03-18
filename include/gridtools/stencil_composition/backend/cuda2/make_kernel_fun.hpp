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

#include "../../../common/defs.hpp"
#include "../../../common/generic_metafunctions/utility.hpp"
#include "../../../common/gt_math.hpp"
#include "../../../common/host_device.hpp"
#include "../../../common/tuple_util.hpp"
#include "../../../meta.hpp"
#include "../../../sid/blocked_dim.hpp"
#include "../../../sid/concept.hpp"
#include "../../be_api.hpp"
#include "../../common/dim.hpp"
#include "j_cache.hpp"

namespace gridtools {
    namespace cuda2 {
        namespace make_kernel_fun_impl_ {
            template <class Deref, int_t JBlockSize>
            struct j_loop_f {
                template <class Info, class Ptr, class Strides, class Validator>
                GT_FUNCTION_DEVICE void operator()(
                    Info, Ptr ptr, Strides const &strides, Validator const &validator) const {
                    using namespace literals;
                    constexpr auto step = 1_c;
                    using max_extent_t = typename Info::extent_t;

                    sid::shift(ptr, sid::get_stride<dim::j>(strides), max_extent_t::minus(dim::j()));

                    using j_caches_t = j_caches_type<Info>;
                    j_caches_t j_caches;
                    auto mixed_ptr = hymap::device::merge(j_caches.ptr(), wstd::move(ptr));

                    auto shift_mixed_ptr = [&](auto dim, auto offset) {
                        sid::shift(mixed_ptr.secondary(), sid::get_stride<decltype(dim)>(strides), offset);
                        j_caches_t::shift(dim, mixed_ptr.primary(), offset);
                    };

#pragma unroll
                    for (int_t j = max_extent_t::jminus::value - max_extent_t::jplus::value; j < JBlockSize; ++j) {
                        device::for_each<typename Info::cells_t>([&](auto cell) GT_FORCE_INLINE_LAMBDA {
                            using cell_t = decltype(cell);
                            using extent_t = typename cell_t::extent_t;
                            constexpr auto j_offset = typename max_extent_t::jplus() - typename extent_t::jplus();
                            shift_mixed_ptr(dim::j(), j_offset);
                            shift_mixed_ptr(dim::i(), typename extent_t::iminus());
#pragma unroll
                            for (int_t i = extent_t::iminus::value; i <= extent_t::iplus::value; ++i) {
                                if (validator(extent_t(), i, j + j_offset))
                                    cell.template operator()<Deref>(mixed_ptr, strides);
                                shift_mixed_ptr(dim::i(), step);
                            }
                            shift_mixed_ptr(dim::i(), -typename extent_t::iplus() - step);
                            shift_mixed_ptr(dim::j(), -j_offset);
                        });
                        sid::shift(mixed_ptr.secondary(), sid::get_stride<dim::j>(strides), step);
                        j_caches.slide();
                    }
                }
            };

            template <class JLoop, class Mss, class Sizes, int_t KBlockSize>
            struct k_loop_f {
                Sizes m_sizes;

                template <class Ptr, class Strides, class Validator>
                GT_FUNCTION_DEVICE void operator()(Ptr ptr, Strides const &strides, Validator validator) const {
                    int_t cur = -(int_t)blockIdx.z * KBlockSize;
                    sid::shift(ptr, sid::get_stride<dim::k>(strides), -cur);
                    tuple_util::device::for_each(
                        [&](int_t size, auto info) GT_FORCE_INLINE_LAMBDA {
                            if (cur >= KBlockSize)
                                return;
                            int_t lim = math::min(cur + size, KBlockSize) - math::max(cur, 0);
                            cur += size;
#pragma unroll
                            for (int_t i = 0; i < KBlockSize; ++i) {
                                if (i >= lim)
                                    break;
                                JLoop()(info, ptr, strides, validator);
                                info.inc_k(ptr, strides);
                            }
                        },
                        m_sizes,
                        Mss::interval_infos());
                }
            };

            template <class Sid, class KLoop>
            struct kernel_f {
                sid::ptr_holder_type<Sid> m_ptr_holder;
                sid::strides_type<Sid> m_strides;
                KLoop k_loop;

                template <class Validator>
                GT_FUNCTION_DEVICE void operator()(int_t i_block, Validator validator) const {
                    auto ptr = m_ptr_holder();
                    sid::shift(ptr, sid::get_stride<sid::blocked_dim<dim::i>>(m_strides), blockIdx.x);
                    sid::shift(ptr, sid::get_stride<sid::blocked_dim<dim::j>>(m_strides), blockIdx.y);
                    sid::shift(ptr, sid::get_stride<dim::i>(m_strides), i_block);
                    k_loop(wstd::move(ptr), m_strides, wstd::move(validator));
                }
            };

            template <class Deref, class Mss, int_t JBlockSize, int_t KBlockSize, class Grid, class Composite>
            auto make_kernel_fun(Grid const &grid, Composite &composite) {
                sid::ptr_diff_type<Composite> offset{};
                auto strides = sid::get_strides(composite);
                sid::shift(offset, sid::get_stride<dim::k>(strides), grid.k_start(Mss::interval(), Mss::execution()));
                auto k_sizes = be_api::make_k_sizes(Mss::interval_infos(), grid);
                using k_sizes_t = decltype(k_sizes);
                using k_loop_t = k_loop_f<j_loop_f<Deref, JBlockSize>, Mss, k_sizes_t, KBlockSize>;
                return kernel_f<Composite, k_loop_t>{
                    sid::get_origin(composite) + offset, std::move(strides), {std::move(k_sizes)}};
            }
        } // namespace make_kernel_fun_impl_
        using make_kernel_fun_impl_::make_kernel_fun;
    } // namespace cuda2
} // namespace gridtools

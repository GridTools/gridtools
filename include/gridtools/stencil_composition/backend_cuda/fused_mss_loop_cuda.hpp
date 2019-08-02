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
            struct deref_f {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
                template <class T>
                GT_FUNCTION std::enable_if_t<is_texture_type<T>::value, T> operator()(T const *ptr) const {
                    return __ldg(ptr);
                }
#endif
                template <class Ptr>
                GT_FUNCTION decltype(auto) operator()(Ptr ptr) const {
                    return *ptr;
                }
            };

            GT_FUNCTION_DEVICE void syncthreads(std::true_type) { __syncthreads(); }
            GT_FUNCTION_DEVICE void syncthreads(std::false_type) {}

            template <class Info, class Ptr, class Strides, class Validator>
            GT_FUNCTION_DEVICE void exec_cells(Info,
                Ptr &GT_RESTRICT ptr,
                Strides const &GT_RESTRICT strides,
                Validator const &GT_RESTRICT validator) {
                device::for_each<typename Info::cells_t>([&](auto cell) {
                    syncthreads(cell.need_sync());
                    if (validator(cell.extent()))
                        cell.template operator()<deref_f>(ptr, strides);
                });
            }

            template <class Mss,
                class Sizes,
                class = typename has_k_caches<Mss>::type,
                class = typename Mss::execution_t>
            struct k_loop_f;

            template <class Mss, class Sizes, class Execution>
            struct k_loop_f<Mss, Sizes, std::true_type, Execution> {
                Sizes m_sizes;

                template <class Ptr, class Strides, class Validator>
                GT_FUNCTION_DEVICE void operator()(Ptr ptr, Strides const &strides, Validator validator) const {
                    k_caches_type<Mss> k_caches;
                    auto mixed_ptr = hymap::device::merge(k_caches.ptr(), wstd::move(ptr));
                    tuple_util::device::for_each(
                        [&](int_t size, auto info) {
                            for (int_t i = 0; i < size; ++i) {
                                exec_cells(info, mixed_ptr, strides, validator);
                                k_caches.slide(info.k_step());
                                info.inc_k(mixed_ptr.secondary(), strides);
                            }
                        },
                        m_sizes,
                        Mss::interval_infos());
                }
            };

            template <class Mss, class Sizes, class Execution>
            struct k_loop_f<Mss, Sizes, std::false_type, Execution> {
                Sizes m_sizes;

                template <class Ptr, class Strides, class Validator>
                GT_FUNCTION_DEVICE void operator()(Ptr ptr, Strides const &strides, Validator validator) const {
                    tuple_util::device::for_each(
                        [&](int_t size, auto info) {
                            for (int_t i = 0; i < size; ++i) {
                                exec_cells(info, ptr, strides, validator);
                                info.inc_k(ptr, strides);
                            }
                        },
                        m_sizes,
                        Mss::interval_infos());
                }
            };

            template <class Mss, class Sizes, int_t BlockSize>
            struct k_loop_f<Mss, Sizes, std::false_type, execute::parallel_block<BlockSize>> {
                Sizes m_sizes;

                template <class Ptr, class Strides, class Validator>
                GT_FUNCTION_DEVICE void operator()(Ptr ptr, Strides const &strides, Validator validator) const {
                    int_t cur = -(int_t)blockIdx.z * BlockSize;
                    sid::shift(ptr, sid::get_stride<dim::k>(strides), -cur);
                    tuple_util::device::for_each(
                        [&](int_t size, auto info) {
                            if (cur >= BlockSize)
                                return;
                            int_t lim = math::min(cur + size, BlockSize) - math::max(cur, 0);
                            cur += size;
#pragma unroll BlockSize
                            for (int_t i = 0; i < BlockSize; ++i) {
                                if (i >= lim)
                                    break;
                                exec_cells(info, ptr, strides, validator);
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
                GT_FUNCTION_DEVICE void operator()(int_t i_block, int_t j_block, Validator validator) const {
                    sid::ptr_diff_type<Sid> offset{};
                    sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::i>>(m_strides), blockIdx.x);
                    sid::shift(offset, sid::get_stride<sid::blocked_dim<dim::j>>(m_strides), blockIdx.y);
                    sid::shift(offset, sid::get_stride<dim::i>(m_strides), i_block);
                    sid::shift(offset, sid::get_stride<dim::j>(m_strides), j_block);
                    k_loop(m_ptr_holder() + offset, m_strides, wstd::move(validator));
                }
            };

            template <class Mss, class Grid, class Composite>
            auto make_kernel_fun(Grid const &grid, Composite &composite) {
                sid::ptr_diff_type<Composite> offset{};
                auto strides = sid::get_strides(composite);
                sid::shift(offset, sid::get_stride<dim::k>(strides), grid.k_start(Mss::interval(), Mss::execution()));
                auto origin = sid::get_origin(composite) + offset;
                auto k_sizes = stage_matrix::make_k_sizes(Mss::interval_infos(), grid);
                using k_sizes_t = decltype(k_sizes);
                using k_loop_t = k_loop_f<Mss, k_sizes_t>;

                return kernel_f<Composite, k_loop_t>{
                    sid::get_origin(composite) + offset, std::move(strides), {std::move(k_sizes)}};
            }
        } // namespace fused_mss_loop_cuda_impl_
        using fused_mss_loop_cuda_impl_::make_kernel_fun;
    } // namespace cuda
} // namespace gridtools

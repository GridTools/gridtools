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
#include "../../common/host_device.hpp"
#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
#include "../caches/extract_extent_caches.hpp"
#include "../dim.hpp"
#include "../execution_types.hpp"
#include "../sid/synthetic.hpp"

namespace gridtools {
    namespace cuda {
        namespace k_cache_impl_ {
            template <class T>
            struct ptr_holder {
                GT_FUNCTION_DEVICE T *operator()() const { return nullptr; }
            };

            template <class T, class U>
            ptr_holder<T> constexpr operator+(ptr_holder<T> obj, U &&) {
                return {};
            }

            template <class ExecutoinType>
            struct k_step : integral_constant<int_t, 1> {};

            template <>
            struct k_step<execute::backward> : integral_constant<int_t, -1> {};

            template <int_t Minus, int_t Plus>
            struct stride {
                static constexpr int_t minus = Minus;
                static constexpr int_t plus = Plus;
                static constexpr int_t size = Plus - Minus + 1;
            };

            template <class T, class ExecutoinType, int_t Minus, int_t Plus>
            GT_FUNCTION void sid_shift(T *p, stride<Minus, Plus>, k_step<ExecutoinType>) {
#pragma unroll
                for (int_t k = Minus; k < Plus; ++k)
                    p[k] = p[k + 1];
            }

            template <class T, int_t Minus, int_t Plus>
            GT_FUNCTION void sid_shift(T *p, stride<Minus, Plus>, k_step<execute::backward>) {
#pragma unroll
                for (int_t k = Plus; k > Minus; --k)
                    p[k] = p[k - 1];
            }

            template <class T, int_t Minus, int_t Plus, class Offset>
            GT_FUNCTION void sid_shift(T &p, stride<Minus, Plus>, Offset offset) {
                p += offset;
            }

            template <class Ptr, class Stride>
            struct k_cache_holder {};

            template <class T, int_t Minus, int_t Plus>
            struct k_cache_holder<T *, stride<Minus, Plus>> {
                T m_values[Plus - Minus + 1];
            };

            template <class Ptr, class Stride>
            GT_FUNCTION_DEVICE void bind_k_cache(k_cache_holder<Ptr, Stride> &, Ptr &, Stride const &) {}

            template <class T, int_t Minus, int_t Plus>
            GT_FUNCTION_DEVICE void bind_k_cache(
                k_cache_holder<T *, stride<Minus, Plus>> &holder, T *&ptr, stride<Minus, Plus>) {
                ptr = holder.m_values - Minus;
            }

            template <class Ptrs,
                class Strides,
                class KStrides =
                    std::decay_t<decltype(::gridtools::sid::get_stride<dim::k>(std::declval<Strides const &>()))>>
            using k_caches_holder = meta::rename<tuple, meta::transform<k_cache_holder, Ptrs, KStrides>>;

            template <class Ptrs, class Strides>
            GT_FUNCTION_DEVICE void bind_k_caches(
                k_caches_holder<Ptrs, Strides> &holder, Ptrs &ptrs, Strides const &strides) {
                tuple_util::device::for_each(
                    [](auto &holder, auto &ptr, auto const &stride) { bind_k_cache(holder, ptr, stride); },
                    holder,
                    ptrs,
                    sid::get_stride<dim::k>(strides));
            }

            template <class Plh, class Esfs>
            auto make_k_cache(Plh, Esfs) {
                using extent_t = extract_k_extent_for_cache<Plh, Esfs>;
                return sid::synthetic()
                    .set<sid::property::ptr_diff, int_t>()
                    .set<sid::property::strides>(
                        hymap::keys<dim::k>::values<stride<extent_t::kminus::value, extent_t::kplus::value>>())
                    .template set<sid::property::origin>(
                        k_cache_impl_::ptr_holder<typename Plh::data_store_t::data_t>());
            }

            template <class>
            struct k_cache_original {};
        } // namespace k_cache_impl_

        using k_cache_impl_::bind_k_caches;
        using k_cache_impl_::k_cache_original;
        using k_cache_impl_::k_caches_holder;
        using k_cache_impl_::k_step;
        using k_cache_impl_::make_k_cache;
    } // namespace cuda
} // namespace gridtools

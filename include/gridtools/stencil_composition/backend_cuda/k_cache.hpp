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
#include "../caches/cache_traits.hpp"
#include "../caches/extract_extent_caches.hpp"
#include "../dim.hpp"
#include "../execution_types.hpp"
#include "../sid/synthetic.hpp"

namespace gridtools {
    namespace cuda {
        namespace k_cache_impl_ {
            template <class T>
            struct ptr {
                T *m_ptr;
                GT_FUNCTION T &operator*() const { return *m_ptr; }
            };

            template <class T, class Stride>
            GT_FUNCTION void sid_shift(ptr<T> &p, Stride &&, int_t offset) {
                p.m_ptr += offset;
            }

            template <class T, int_t Minus, int_t Plus>
            struct storage {
                T m_values[Plus - Minus + 1];

                storage() = default;
                storage(storage const &) = delete;
                storage(storage &&) = default;

                template <class Step, std::enable_if_t<Step::value == 1, int> = 0>
                GT_FUNCTION_DEVICE void slide(Step) {
#pragma unroll
                    for (int_t k = 0; k < Plus - Minus; ++k)
                        m_values[k] = m_values[k + 1];
                }

                template <class Step, std::enable_if_t<Step::value == -1, int> = 0>
                GT_FUNCTION_DEVICE void slide(Step) {
#pragma unroll
                    for (int_t k = Plus - Minus; k > 0; --k)
                        m_values[k] = m_values[k - 1];
                }

                GT_FUNCTION_DEVICE ptr<T> ptr() { return {m_values - Minus}; }
            };

            template <class Plh, class T, int_t Minus, int_t Plus>
            struct sync_storage : storage<T, Minus, Plus> {
                int_t m_lower_bound;
                int_t m_upper_bound;

                template <class Step, class Ptrs, class Strides, std::enable_if_t<Step::value == 1, int> = 0>
                GT_FUNCTION_DEVICE void fill(Step, Ptrs const &ptrs, Strides const &strides) {
                    int_t cur = *device::at_key<positional<dim::k>>(ptrs);
                    if (cur + Plus >= m_upper_bound)
                        return;
                    auto ptr = device::at_key<Plh>(ptrs);
                    sid::shift(ptr, sid::get_stride_element<Plh, dim::k>(strides), integral_constant<int_t, Plus>());
                    this->m_values[Plus - Minus] = *ptr;
                }

                template <class Step, class Ptrs, class Strides, std::enable_if_t<Step::value == -1, int> = 0>
                GT_FUNCTION_DEVICE void fill(Step, Ptrs const &ptrs, Strides const &strides) {
                    int_t cur = *device::at_key<positional<dim::k>>(ptrs);
                    if (cur + Minus < m_lower_bound)
                        return;
                    auto ptr = device::at_key<Plh>(ptrs);
                    sid::shift(ptr, sid::get_stride_element<Plh, dim::k>(strides), integral_constant<int_t, Minus>());
                    this->m_values[0] = *ptr;
                }

                template <class Step, class Ptrs, class Strides, std::enable_if_t<Step::value == 1, int> = 0>
                GT_FUNCTION_DEVICE void initial_fill(Step, Ptrs const &ptrs, Strides const &strides) {
                    auto ptr = device::at_key<Plh>(ptrs);
                    sid::shift(ptr, sid::get_stride_element<Plh, dim::k>(strides), integral_constant<int_t, Plus>());
                    int_t begin = m_lower_bound - *device::at_key<positional<dim::k>>(ptrs) - Minus;
#pragma unroll
                    for (int_t k = Plus - Minus; k >= 0; --k) {
                        if (k < begin)
                            return;
                        this->m_values[k] = *ptr;
                        sid::shift(ptr, sid::get_stride_element<Plh, dim::k>(strides), integral_constant<int_t, -1>());
                    }
                }

                template <class Step, class Ptrs, class Strides, std::enable_if_t<Step::value == -1, int> = 0>
                GT_FUNCTION_DEVICE void initial_fill(Step, Ptrs const &ptrs, Strides const &strides) {
                    auto ptr = device::at_key<Plh>(ptrs);
                    sid::shift(ptr, sid::get_stride_element<Plh, dim::k>(strides), integral_constant<int_t, Minus>());
                    int_t end = m_upper_bound - *device::at_key<positional<dim::k>>(ptrs) - Plus;
#pragma unroll
                    for (int_t k = 0; k <= Plus - Minus; ++k) {
                        if (k >= end)
                            return;
                        this->m_values[k] = *ptr;
                        sid::shift(ptr, sid::get_stride_element<Plh, dim::k>(strides), integral_constant<int_t, 1>());
                    }
                }
            };

            struct fake {
                GT_FUNCTION fake operator()() const { return {}; }
                fake operator*() const;
            };
            fake sid_get_ptr_diff(fake);
            fake sid_get_origin(fake) { return {}; }
            GT_FUNCTION void sid_shift(fake &, fake, int_t) {}
            GT_FUNCTION fake operator+(fake, fake) { return {}; }
            hymap::keys<dim::k>::values<fake> sid_get_strides(fake) { return {}; }

            GT_STATIC_ASSERT(is_sid<fake>(), GT_INTERNAL_ERROR);

            template <class Mss>
            auto make_k_cached_sids() {
                using plhs_t =
                    meta::transform<cache_parameter, meta::filter<is_k_cache, typename Mss::cache_sequence_t>>;
                return hymap::from_keys_values<plhs_t, meta::repeat<meta::length<plhs_t>, fake>>();
            }

            template <class Storages>
            class k_caches {
                Storages m_storages;

              public:
                GT_FUNCTION_DEVICE auto ptr() {
                    return tuple_util::device::transform([](auto &storage) { return storage.ptr(); }, m_storages);
                }

                template <class Step>
                GT_FUNCTION_DEVICE void slide(Step step) {
                    tuple_util::device::for_each([step](auto &storage) { storage.slide(step); }, m_storages);
                }
            };

            template <class Esfs>
            struct storage_type_f {
                template <class Plh, class Extent = extract_k_extent_for_cache<Plh, Esfs>>
                using apply = storage<typename Plh::data_store_t::data_t, Extent::kminus::value, Extent::kplus::value>;
            };

            template <class Mss,
                class Plhs = meta::transform<cache_parameter, meta::filter<is_k_cache, typename Mss::cache_sequence_t>>,
                class Storages = meta::transform<storage_type_f<typename Mss::esf_sequence_t>::template apply, Plhs>>
            using k_caches_type = k_caches<hymap::from_keys_values<Plhs, Storages>>;
        } // namespace k_cache_impl_

        using k_cache_impl_::k_caches_type;
        using k_cache_impl_::make_k_cached_sids;
    } // namespace cuda
} // namespace gridtools
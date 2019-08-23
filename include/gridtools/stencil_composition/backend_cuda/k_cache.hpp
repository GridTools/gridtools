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

            template <class Storages>
            class k_caches {
                Storages m_storages;

              public:
                GT_FUNCTION_DEVICE auto ptr() {
                    return tuple_util::device::transform(
                        [](auto &storage) GT_FORCE_INLINE_LAMBDA { return storage.ptr(); }, m_storages);
                }

                template <class Step>
                GT_FUNCTION_DEVICE void slide(Step step) {
                    tuple_util::device::for_each(
                        [step](auto &storage) GT_FORCE_INLINE_LAMBDA { storage.slide(step); }, m_storages);
                }
            };

            template <class PlhInfo, class Extent = typename PlhInfo::extent_t>
            using make_storage_type = storage<typename PlhInfo::data_t, Extent::kminus::value, Extent::kplus::value>;

            template <class PlhInfo>
            using is_k_cached =
                std::is_same<typename PlhInfo::caches_t, meta::list<integral_constant<cache_type, cache_type::k>>>;

            template <class Mss>
            using has_k_caches = meta::any_of<is_k_cached, typename Mss::plh_map_t>;

            template <class Mss,
                class PlhMap = meta::filter<is_k_cached, typename Mss::plh_map_t>,
                class Keys = meta::transform<meta::first, PlhMap>,
                class Storages = meta::transform<make_storage_type, PlhMap>>
            using k_caches_type = k_caches<hymap::from_keys_values<Keys, Storages>>;
        } // namespace k_cache_impl_

        using k_cache_sid_t = k_cache_impl_::fake;
        using k_cache_impl_::has_k_caches;
        using k_cache_impl_::k_caches_type;
    } // namespace cuda
} // namespace gridtools

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

#include "../../../common/defs.hpp"
#include "../../../common/host_device.hpp"
#include "../../../common/hymap.hpp"
#include "../../../common/integral_constant.hpp"
#include "../../common/caches.hpp"
#include "../../common/dim.hpp"
#include "../../common/extent.hpp"

namespace gridtools {
    namespace cuda2 {
        namespace j_cache_impl_ {

            template <class Extent>
            using ij_strides_t = hymap::keys<dim::i, dim::j>::values<integral_constant<int_t, 1>,
                integral_constant<int_t, Extent::iplus::value - Extent::iminus::value + 1>>;

            template <class T, class Extent>
            struct storage;
            template <class T, int_t IMinus, int_t IPlus, int_t JMinus, int_t JPlus>
            struct storage<T, extent<IMinus, IPlus, JMinus, JPlus>> {
                T m_values[IPlus - IMinus + 1][JPlus - JMinus + 1];
                storage() = default;
                storage(storage const &) = delete;
                storage(storage &&) = default;

                GT_FUNCTION_DEVICE T *ptr() { return &m_values[-IMinus][-JMinus]; }

                GT_FUNCTION_DEVICE void slide() {
#pragma unroll
                    for (int_t j = 0; j < JPlus - JMinus; ++j)
#pragma unroll
                        for (int_t i = 0; i < IPlus - IMinus + 1; ++i)
                            m_values[i][j] = m_values[i][j - 1];
                }

                using strides_t = ij_strides_t<extent<IMinus, IPlus, JMinus, JPlus>>;
            };

            template <class T>
            using get_strides = typename T::strides_t;

            template <class Dim>
            struct get_stride_f {
                template <class T>
                using apply = std::decay_t<decltype(at_key<Dim>(typename T::strides_t()))>;
            };

            template <class Extent>
            struct fake {
                GT_FUNCTION fake operator()() const { return {}; }
                fake operator*() const;
            };
            template <class Extent>
            fake<Extent> sid_get_ptr_diff(fake<Extent>);

            template <class Extent>
            fake<Extent> sid_get_origin(fake<Extent>) {
                return {};
            }

            template <class Extent>
            GT_FUNCTION fake<Extent> operator+(fake<Extent>, fake<Extent>) {
                return {};
            }

            template <class Extent>
            ij_strides_t<Extent> sid_get_strides(fake<Extent>) {
                return {};
            }

            template <class Storages>
            class j_caches {
                Storages m_storages;

              public:
                GT_FUNCTION_DEVICE auto ptr() {
                    return tuple_util::device::transform(
                        [](auto &storage) GT_FORCE_INLINE_LAMBDA { return storage.ptr(); }, m_storages);
                }

                GT_FUNCTION_DEVICE void slide() {
                    tuple_util::device::for_each(
                        [](auto &storage) GT_FORCE_INLINE_LAMBDA { storage.slide(); }, m_storages);
                }

                template <class Ptr, class Dim, class Offset>
                static GT_FUNCTION_DEVICE void shift(Dim, Ptr &ptr, Offset offset) {
                    using strides_t = meta::rename<tuple, meta::transform<get_stride_f<Dim>::apply, Storages>>;
                    tuple_util::device::for_each(
                        [offset](auto &ptr, auto stride) { sid::shift(ptr, stride, offset); }, ptr, strides_t());
                }
            };

            template <class PlhInfo>
            using make_storage_type = storage<typename PlhInfo::data_t, typename PlhInfo::extent_t>;

            template <class Info,
                class PlhMap = meta::filter<be_api::get_is_tmp, typename Info::plh_map_t>,
                class Keys = meta::transform<meta::first, PlhMap>,
                class Storages = meta::transform<make_storage_type, PlhMap>>
            using j_caches_type = j_caches<hymap::from_keys_values<Keys, Storages>>;
        } // namespace j_cache_impl_

        template <class Extent>
        using j_cache_sid_t = j_cache_impl_::fake<Extent>;
        //        using j_cache_impl_::has_k_caches;
        using j_cache_impl_::j_caches_type;
    } // namespace cuda2
} // namespace gridtools

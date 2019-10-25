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

#include "../meta/id.hpp"
#include "common/definitions.hpp"
#include "data_store.hpp"

namespace gridtools {
    namespace data_view_impl_ {
        template <class T, class StorageInfo>
        struct host_view {
            T *m_ptr;
            StorageInfo const *m_info;

            using storage_info_t = StorageInfo;
            using data_t = T;

            T *data() const { return m_ptr; }

            StorageInfo const &storage_info() const { return *m_info; }

            template <typename... Coords>
            T &operator()(Coords... c) const {
                static_assert(conjunction<is_all_integral_or_enum<Coords...>>::value,
                    GT_INTERNAL_ERROR_MSG("Index arguments have to be integral types."));
                return m_ptr[m_info->index(c...)];
            }

            T &operator()(array<int, StorageInfo::ndims> const &arr) const { return m_ptr[m_info->index(arr)]; }

            constexpr auto padded_total_length() const { return m_info->padded_total_length(); }
            template <uint_t Dim>
            constexpr auto length() const {
                return m_info->template length<Dim>();
            }
            template <uint_t Dim>
            constexpr auto total_length() const {
                return m_info->template total_length<Dim>();
            }
            template <uint_t Dim>
            constexpr auto total_begin() const {
                return m_info->template total_begin<Dim>();
            }
            template <uint_t Dim>
            constexpr auto begin() const {
                return m_info->template begin<Dim>();
            }
            template <uint_t Dim>
            constexpr auto total_end() const {
                return m_info->template total_end<Dim>();
            }
            template <uint_t Dim>
            constexpr auto end() const {
                return m_info->template end<Dim>();
            }

            friend T *advanced_get_raw_pointer_of(host_view const &src) { return src.m_ptr; }
        };

        template <class Storage, class Info, class T>
        bool check_consistency_impl(Storage &storage, host_view<T, Info> const &view) {
            return view.data() == storage.get_cpu_ptr();
        }

        template <class Storage,
            class StorageInfo,
            class View,
            class StorageData = typename Storage::data_t,
            class ViewData = typename View::data_t>
        std::enable_if_t<std::is_same<StorageData, ViewData>::value ||
                             std::is_same<StorageData, std::remove_const_t<ViewData>>::value,
            bool>
        check_consistency(data_store<Storage, StorageInfo> const &d, View const &view) {
            return check_consistency_impl(d.storage(), view);
        }

        template <class Mode, class Type, class Storage, class Info>
        host_view<typename Type::type, Info> make_host_view_impl(Mode, Type, Storage &storage, Info const &info) {
            return {storage.get_cpu_ptr(), &info};
        }

        template <class Mode, class Type, class Storage, class Info>
        auto make_target_view_impl(Mode mode, Type type, Storage &storage, Info const &info) {
            return make_host_view_impl(mode, type, storage, info);
        }

        template <access_mode Mode = access_mode::read_write,
            class Storage,
            class Info,
            class Data = typename Storage::data_t>
        auto make_host_view(data_store<Storage, Info> const &ds) {
            return make_host_view_impl(access_mode_type<Mode, Data>(),
                meta::lazy::id<apply_access_mode<Mode, Data>>(),
                ds.storage(),
                ds.info());
        }

        template <access_mode Mode = access_mode::read_write,
            class Storage,
            class Info,
            class Data = typename Storage::data_t>
        auto make_target_view(data_store<Storage, Info> const &ds) {
            return make_target_view_impl(access_mode_type<Mode, Data>(),
                meta::lazy::id<apply_access_mode<Mode, Data>>(),
                ds.storage(),
                ds.info());
        }
    } // namespace data_view_impl_

    using data_view_impl_::check_consistency;
    using data_view_impl_::host_view;
    using data_view_impl_::make_host_view;
    using data_view_impl_::make_target_view;
} // namespace gridtools

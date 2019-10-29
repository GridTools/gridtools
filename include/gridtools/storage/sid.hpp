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

#include <cassert>

#include "../common/defs.hpp"
#include "../common/host_device.hpp"
#include "../common/hymap.hpp"
#include "../common/integral_constant.hpp"
#include "../common/layout_map.hpp"
#include "../common/tuple.hpp"
#include "../common/tuple_util.hpp"
#include "../meta/if.hpp"
#include "../meta/macros.hpp"
#include "../meta/make_indices.hpp"
#include "../meta/transform.hpp"
#include "data_store.hpp"

namespace gridtools {
    namespace storage_sid_impl_ {

        enum class dimension_kind { masked, innermost, dynamic };

        constexpr dimension_kind get_dimension_kind(int i, size_t n_dim) {
            return i < 0 ? dimension_kind::masked
                         : i + 1 == n_dim ? dimension_kind::innermost : dimension_kind::dynamic;
        }

        template <dimension_kind Kind>
        struct stride_type;

        template <>
        struct stride_type<dimension_kind::masked> {
            using type = integral_constant<int_t, 0>;
        };

        template <>
        struct stride_type<dimension_kind::innermost> {
            using type = integral_constant<int_t, 1>;
        };

        template <>
        struct stride_type<dimension_kind::dynamic> {
            using type = int_t;
        };

        template <class I, class Res>
        struct stride_generator_f;

        template <class I, int V>
        struct stride_generator_f<I, integral_constant<int_t, V>> {
            using type = stride_generator_f;
            template <class Src>
            integral_constant<int_t, V> operator()(Src const &src) {
                assert(src[I::value] == V);
                return {};
            }
        };

        template <class I>
        struct stride_generator_f<I, int_t> {
            using type = stride_generator_f;
            template <class Src>
            int_t operator()(Src const &src) {
                assert(src[I::value] != 0);
                return (int_t)src[I::value];
            }
        };

        template <class Layout>
        struct convert_strides_f;

        template <int... Is>
        struct convert_strides_f<layout_map<Is...>> {
            using res_t =
                tuple<typename stride_type<get_dimension_kind(Is, layout_map<Is...>::unmasked_length)>::type...>;
            using generators_t = meta::transform<stride_generator_f, meta::make_indices_c<sizeof...(Is)>, res_t>;

            template <class Src>
            res_t operator()(Src const &src) const {
                return tuple_util::generate<generators_t, res_t>(src);
            }
        };

        struct empty_ptr_diff {
            template <class T>
            friend GT_CONSTEXPR GT_FUNCTION T *operator+(T *lhs, empty_ptr_diff) {
                return lhs;
            }
        };

        template <class T>
        struct ptr_holder {
            T *m_val;
            GT_FUNCTION GT_CONSTEXPR T *operator()() const { return m_val; }

            friend GT_FORCE_INLINE constexpr ptr_holder operator+(ptr_holder obj, int_t arg) {
                return {obj.m_val + arg};
            }

            friend GT_FORCE_INLINE constexpr ptr_holder operator+(ptr_holder obj, empty_ptr_diff) { return obj; }
        };

        template <class Storage, class StorageInfo>
        class host_adapter {
            using impl_t = data_store<Storage, StorageInfo>;
            impl_t m_impl;

            static impl_t const &impl();

            friend ptr_holder<typename Storage::data_t> sid_get_origin(host_adapter const &obj) {
                auto &&storage = obj.m_impl.storage();
                if (storage.host_needs_update())
                    storage.sync();
                storage.reactivate_host_write_views();
                return {storage.get_cpu_ptr()};
            }
            friend decltype(sid_get_strides(impl())) sid_get_strides(host_adapter const &obj) {
                return sid_get_strides(obj.m_impl);
            }
            friend decltype(sid_get_lower_bounds(impl())) sid_get_lower_bounds(host_adapter const &obj) {
                return sid_get_lower_bounds(obj.m_impl);
            }
            friend decltype(sid_get_upper_bounds(impl())) sid_get_upper_bounds(host_adapter const &obj) {
                return sid_get_upper_bounds(obj.m_impl);
            }
            friend decltype(sid_get_strides_kind(impl())) sid_get_strides_kind(host_adapter const &) { return {}; }
            friend decltype(sid_get_ptr_diff(impl())) sid_get_ptr_diff(host_adapter const &) { return {}; }

          public:
            host_adapter(data_store<Storage, StorageInfo> obj) : m_impl(std::move(obj)) {}
        };

        template <class Src>
        using to_dim = integral_constant<int_t, Src::value>;

        namespace lazy {
            template <class Layout>
            struct get_unmasked_dims;
            template <int... Is>
            struct get_unmasked_dims<layout_map<Is...>> {
                using indices_t = meta::make_indices_c<sizeof...(Is), tuple>;
                using dims_t = meta::transform<to_dim, indices_t>;
                using items_t = meta::zip<dims_t, meta::list<bool_constant<Is >= 0>...>>;
                using filtered_items_t = meta::filter<meta::second, items_t>;
                using type = meta::transform<meta::first, filtered_items_t>;
            };
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(get_unmasked_dims, class Layout, Layout);

        template <class StorageInfo,
            class Value,
            class Layout = typename StorageInfo::layout_t,
            class Dims = get_unmasked_dims<Layout>,
            class Values = meta::repeat_c<Layout::unmasked_length, Value>>
        using bounds_type = hymap::from_keys_values<Dims, Values>;

        template <class Dim>
        struct upper_bound_generator_f {
            using type = upper_bound_generator_f;

            template <class StorageInfo>
            int_t operator()(StorageInfo const &info) const {
                return info.lengths()[Dim::value];
            }
        };

    } // namespace storage_sid_impl_

    /**
     *   The functions below make `data_store` model the `SID` concept
     */
    template <class Storage, class StorageInfo>
    storage_sid_impl_::ptr_holder<typename Storage::data_t> sid_get_origin(
        data_store<Storage, StorageInfo> const &obj) {
        auto &&storage = obj.storage();
        if (storage.device_needs_update())
            storage.sync();
        storage.reactivate_target_write_views();
        return {storage.get_target_ptr()};
    }

    template <class Storage, class StorageInfo>
    auto sid_get_strides(data_store<Storage, StorageInfo> const &obj) {
        return storage_sid_impl_::convert_strides_f<typename StorageInfo::layout_t>{}(obj.strides());
    }

    template <class Storage, class StorageInfo>
    StorageInfo sid_get_strides_kind(data_store<Storage, StorageInfo> const &);

    template <class Storage, class StorageInfo>
    meta::if_c<StorageInfo::layout_t::unmasked_length == 0, storage_sid_impl_::empty_ptr_diff, int_t> sid_get_ptr_diff(
        data_store<Storage, StorageInfo> const &);

    template <class Storage, class StorageInfo>
    storage_sid_impl_::bounds_type<StorageInfo, integral_constant<int_t, 0>> sid_get_lower_bounds(
        data_store<Storage, StorageInfo> const &) {
        return {};
    }

    template <class Storage, class StorageInfo, class Res = storage_sid_impl_::bounds_type<StorageInfo, int_t>>
    Res sid_get_upper_bounds(data_store<Storage, StorageInfo> const &obj) {
        using keys_t = get_keys<Res>;
        using generators_t = meta::transform<storage_sid_impl_::upper_bound_generator_f, keys_t>;
        return tuple_util::generate<generators_t, Res>(obj.info());
    }

    /**
     *  Returns an object that models the `SID` concept the same way as original object does except that the
     * pointers are taken from the host view.
     */
    template <class Storage, class StorageInfo>
    storage_sid_impl_::host_adapter<Storage, StorageInfo> as_host(data_store<Storage, StorageInfo> obj) {
        return {std::move(obj)};
    }
} // namespace gridtools

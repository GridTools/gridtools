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
    namespace storage {
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

            template <class DataStore,
                class Value,
                class Layout = typename DataStore::layout_t,
                class Dims = get_unmasked_dims<Layout>,
                class Values = meta::repeat_c<Layout::unmasked_length, Value>>
            using bounds_type = hymap::from_keys_values<Dims, Values>;

            template <class Dim>
            struct upper_bound_generator_f {
                using type = upper_bound_generator_f;

                template <class Lengths>
                int_t operator()(Lengths const &lengths) const {
                    return lengths[Dim::value];
                }
            };

        } // namespace storage_sid_impl_

        /**
         *   The functions below make `data_store` model the `SID` concept
         */
        template <class DataStore, std::enable_if_t<is_data_store<DataStore>::value, int> = 0>
        storage_sid_impl_::ptr_holder<typename DataStore::data_t> sid_get_origin(std::shared_ptr<DataStore> const &ds) {
            return {ds->get_target_ptr()};
        }

        template <class DataStore, std::enable_if_t<is_data_store<DataStore>::value, int> = 0>
        auto sid_get_strides(std::shared_ptr<DataStore> const &ds) {
            return storage_sid_impl_::convert_strides_f<typename DataStore::layout_t>{}(ds->strides());
        }

        template <class DataStore, std::enable_if_t<is_data_store<DataStore>::value, int> = 0>
        typename DataStore::kind_t sid_get_strides_kind(std::shared_ptr<DataStore> const &);

        template <class DataStore, std::enable_if_t<is_data_store<DataStore>::value, int> = 0>
        meta::if_c<DataStore::layout_t::unmasked_length == 0, storage_sid_impl_::empty_ptr_diff, int_t>
        sid_get_ptr_diff(std::shared_ptr<DataStore> const &);

        template <class DataStore, std::enable_if_t<is_data_store<DataStore>::value, int> = 0>
        storage_sid_impl_::bounds_type<DataStore, integral_constant<int_t, 0>> sid_get_lower_bounds(
            std::shared_ptr<DataStore> const &) {
            return {};
        }

        template <class DataStore, std::enable_if_t<is_data_store<DataStore>::value, int> = 0>
        auto sid_get_upper_bounds(std::shared_ptr<DataStore> const &ds) {
            using res_t = storage_sid_impl_::bounds_type<DataStore, int_t>;
            using keys_t = get_keys<res_t>;
            using generators_t = meta::transform<storage_sid_impl_::upper_bound_generator_f, keys_t>;
            return tuple_util::generate<generators_t, res_t>(ds->lengths());
        }
    } // namespace storage
} // namespace gridtools

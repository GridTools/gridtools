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

#include "../common/defs.hpp"
#include "../common/hymap.hpp"
#include "../meta.hpp"
#include "../storage/sid.hpp"
#include "arg.hpp"
#include "caches/cache_traits.hpp"
#include "dim.hpp"
#include "extent.hpp"
#include "extract_placeholders.hpp"
#include "positional.hpp"
#include "sid/composite.hpp"
#include "sid/concept.hpp"

namespace gridtools {

    namespace local_domain_impl_ {

        template <class T>
        struct is_local_domain : std::false_type {};

        namespace lazy {
            template <class Arg>
            struct get_storage {
                using type = typename Arg::data_store_t;
            };

            template <class Dim>
            struct get_storage<positional<Dim>> {
                using type = positional<Dim>;
            };

            template <bool IsStateful, bool NeedK>
            struct positionals : meta::list<positional<dim::i>, positional<dim::j>, positional<dim::k>> {};

            template <>
            struct positionals<false, false> : meta::list<> {};

            template <>
            struct positionals<false, true> : meta::list<positional<dim::k>> {};

            template <class Args, bool IsStateful>
            struct add_positional : meta::lazy::id<Args> {};

            template <class Args>
            struct add_positional<Args, true>
                : meta::lazy::push_back<Args, positional<dim::i>, positional<dim::j>, positional<dim::k>> {};
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(get_storage, class Arg, Arg);
        GT_META_DELEGATE_TO_LAZY(positionals, (bool IsStateful, bool NeedK), (IsStateful, NeedK));
        GT_META_DELEGATE_TO_LAZY(add_positional, (class Args, bool IsStateful), (Args, IsStateful));

        template <class Arg>
        using get_lower_bound = std::decay_t<
            meta::second<meta::mp_find<hymap::to_meta_map<sid::lower_bounds_type<get_storage<Arg>>>, dim::k>>>;

        template <class Arg>
        using get_upper_bound = std::decay_t<
            meta::second<meta::mp_find<hymap::to_meta_map<sid::upper_bounds_type<get_storage<Arg>>>, dim::k>>>;

        template <class Arg>
        using has_k_lower_bound = has_key<sid::lower_bounds_type<get_storage<Arg>>, dim::k>;

        template <class Arg>
        using has_k_upper_bound = has_key<sid::upper_bounds_type<get_storage<Arg>>, dim::k>;

        template <class Arg>
        using get_storage_info = typename Arg::data_store_t::storage_info_t;

        template <class Arg, class Src, class Dst>
        struct set_stride_f {
            Src const &m_src;
            Dst &m_dst;

            template <class Dim>
            void operator()() const {
                at_key<Arg>(at_key<Dim>(m_dst)) = at_key<Dim>(m_src);
            }
        };
        template <class Arg, class Src, class Dst>
        set_stride_f<Arg, Src, Dst> set_stride(Src const &src, Dst &dst) {
            return {src, dst};
        }

        struct sink {
            template <class T>
            sink &operator=(T &&) {
                return *this;
            }
        };

        template <class Backend, class Mss, class MaxExtentForTmp, bool IsStateful>
        struct get_local_domain;

        template <class Composite>
        struct naive_local_domain {
            GT_STATIC_ASSERT(is_sid<Composite>::value, GT_INTERNAL_ERROR);

            using ptr_holder_t = sid::ptr_holder_type<Composite>;
            using strides_t = sid::strides_type<Composite>;

            template <class Arg, class DataStore, std::enable_if_t<has_key<Composite, Arg>::value, int> = 0>
            void set_data_store(Arg, DataStore &data_store) {
                GT_STATIC_ASSERT(is_sid<DataStore>::value, "");

                at_key<Arg>(m_ptr_holder) = sid::get_origin(data_store);
                using stride_dims_t = get_keys<sid::strides_type<DataStore>>;
                auto const &src_strides = sid::get_strides(data_store);
                for_each_type<stride_dims_t>(local_domain_impl_::set_stride<Arg>(src_strides, m_strides));
            }

            template <class Arg, class DataStore, std::enable_if_t<!has_key<Composite, Arg>::value, int> = 0>
            void set_data_store(Arg, DataStore &) {}

            ptr_holder_t m_ptr_holder;
            strides_t m_strides;
        };
        template <class Composite>
        struct is_local_domain<naive_local_domain<Composite>> : std::true_type {};

        template <class Mss, class MaxExtentForTmp, bool IsStateful>
        struct get_local_domain<backend::naive, Mss, MaxExtentForTmp, IsStateful> {
            GT_STATIC_ASSERT(is_extent<MaxExtentForTmp>::value, GT_INTERNAL_ERROR);

            using esf_args_t = extract_placeholders_from_mss<Mss>;
            GT_STATIC_ASSERT((meta::all_of<is_plh, esf_args_t>::value), GT_INTERNAL_ERROR);

            using args_t = meta::push_back<esf_args_t, positional<dim::i>, positional<dim::j>, positional<dim::k>>;

            using composite_keys_t = meta::rename<sid::composite::keys, args_t>;

            using storages_t = meta::transform<get_storage, args_t>;

            using composite_t = meta::rename<composite_keys_t::template values, storages_t>;

            using type = naive_local_domain<composite_t>;
        };

        template <class Composite, class CacheSequence, class TotalLengthMap>
        struct mc_local_domain {
            GT_STATIC_ASSERT(is_sid<Composite>::value, GT_INTERNAL_ERROR);

            using cache_sequence_t = CacheSequence;
            using ptr_holder_t = sid::ptr_holder_type<Composite>;
            using ptr_t = sid::ptr_type<Composite>;
            using strides_t = sid::strides_type<Composite>;

            template <class Arg, class DataStore, std::enable_if_t<has_key<Composite, Arg>::value, int> = 0>
            void set_data_store(Arg, DataStore &data_store) {
                GT_STATIC_ASSERT(is_sid<DataStore>::value, "");

                at_key<Arg>(m_ptr_holder) = sid::get_origin(data_store);
                using stride_dims_t = get_keys<sid::strides_type<DataStore>>;
                auto const &src_strides = sid::get_strides(data_store);
                for_each_type<stride_dims_t>(local_domain_impl_::set_stride<Arg>(src_strides, m_strides));

                at_key_with_default<typename DataStore::storage_info_t, local_domain_impl_::sink>(m_total_length_map) =
                    data_store.info().padded_total_length();
            }

            template <class Arg, class DataStore, std::enable_if_t<!has_key<Composite, Arg>::value, int> = 0>
            void set_data_store(Arg, DataStore &) {}

            ptr_holder_t m_ptr_holder;
            strides_t m_strides;

            TotalLengthMap m_total_length_map;
        };
        template <class Composite, class CacheSequence, class TotalLengthMap>
        struct is_local_domain<mc_local_domain<Composite, CacheSequence, TotalLengthMap>> : std::true_type {};

        template <class Mss, class MaxExtentForTmp, bool IsStateful>
        struct get_local_domain<backend::mc, Mss, MaxExtentForTmp, IsStateful> {
            GT_STATIC_ASSERT(is_extent<MaxExtentForTmp>::value, GT_INTERNAL_ERROR);

            using esf_args_t = extract_placeholders_from_mss<Mss>;
            GT_STATIC_ASSERT((meta::all_of<is_plh, esf_args_t>::value), GT_INTERNAL_ERROR);

            using total_length_esf_args_t = meta::filter<is_tmp_arg, esf_args_t>;

            using total_length_storage_infos_t =
                meta::dedup<meta::transform<local_domain_impl_::get_storage_info, total_length_esf_args_t>>;

            using total_length_map_t = hymap::from_keys_values<total_length_storage_infos_t,
                meta::repeat<meta::length<total_length_storage_infos_t>, uint_t>>;

            using positionals_t = local_domain_impl_::positionals<IsStateful, false>;

            using args_t = meta::concat<esf_args_t, positionals_t>;

            using composite_keys_t = meta::rename<sid::composite::keys, args_t>;

            using storages_t = meta::transform<local_domain_impl_::get_storage, args_t>;

            using composite_t = meta::rename<composite_keys_t::template values, storages_t>;

            using type = mc_local_domain<composite_t, typename Mss::cache_sequence_t, total_length_map_t>;
        };

        template <class Composite, class MaxExtentForTmp>
        struct x86_local_domain {
            GT_STATIC_ASSERT(is_sid<Composite>::value, GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT(is_extent<MaxExtentForTmp>::value, GT_INTERNAL_ERROR);

            using max_extent_for_tmp_t = MaxExtentForTmp;

            using ptr_holder_t = sid::ptr_holder_type<Composite>;
            using ptr_t = sid::ptr_type<Composite>;
            using strides_t = sid::strides_type<Composite>;
            using ptr_diff_t = sid::ptr_diff_type<Composite>;

            template <class Arg, class DataStore, std::enable_if_t<has_key<Composite, Arg>::value, int> = 0>
            void set_data_store(Arg, DataStore &data_store) {
                GT_STATIC_ASSERT(is_sid<DataStore>::value, "");

                at_key<Arg>(m_ptr_holder) = sid::get_origin(data_store);
                using stride_dims_t = get_keys<sid::strides_type<DataStore>>;
                auto const &src_strides = sid::get_strides(data_store);
                for_each_type<stride_dims_t>(local_domain_impl_::set_stride<Arg>(src_strides, m_strides));
            }

            template <class Arg, class DataStore, std::enable_if_t<!has_key<Composite, Arg>::value, int> = 0>
            void set_data_store(Arg, DataStore &) {}

            ptr_holder_t m_ptr_holder;
            strides_t m_strides;
        };
        template <class Composite, class MaxExtentForTmp>
        struct is_local_domain<x86_local_domain<Composite, MaxExtentForTmp>> : std::true_type {};

        template <class Mss, class MaxExtentForTmp, bool IsStateful>
        struct get_local_domain<backend::x86, Mss, MaxExtentForTmp, IsStateful> {
            GT_STATIC_ASSERT(is_extent<MaxExtentForTmp>::value, GT_INTERNAL_ERROR);

            using esf_args_t = extract_placeholders_from_mss<Mss>;
            GT_STATIC_ASSERT((meta::all_of<is_plh, esf_args_t>::value), GT_INTERNAL_ERROR);

            using positionals_t = local_domain_impl_::positionals<IsStateful, false>;

            using args_t = meta::concat<esf_args_t, positionals_t>;

            using composite_keys_t = meta::rename<sid::composite::keys, args_t>;

            using storages_t = meta::transform<local_domain_impl_::get_storage, args_t>;

            using composite_t = meta::rename<composite_keys_t::template values, storages_t>;

            using type = x86_local_domain<composite_t, MaxExtentForTmp>;
        };

        template <class Composite,
            class MaxExtentForTmp,
            class CacheSequence,
            class KLowerBoundsMap,
            class KUpperBoundsMap>
        class cuda_local_domain {
            GT_STATIC_ASSERT(is_extent<MaxExtentForTmp>::value, GT_INTERNAL_ERROR);

            KLowerBoundsMap m_k_lower_bounds_map;
            KUpperBoundsMap m_k_upper_bounds_map;

            template <class Arg, class DataStore, std::enable_if_t<has_key<KLowerBoundsMap, Arg>::value, int> = 0>
            void set_k_lower_bounds(Arg, DataStore const &data_store) {
                at_key<Arg>(m_k_lower_bounds_map) = at_key<dim::k>(sid::get_lower_bounds(data_store));
            }
            template <class Arg, class DataStore, std::enable_if_t<!has_key<KLowerBoundsMap, Arg>::value, int> = 0>
            void set_k_lower_bounds(Arg, DataStore const &data_store) {}

            template <class Arg, class DataStore, std::enable_if_t<has_key<KUpperBoundsMap, Arg>::value, int> = 0>
            void set_k_upper_bounds(Arg, DataStore const &data_store) {
                at_key<Arg>(m_k_upper_bounds_map) = at_key<dim::k>(sid::get_upper_bounds(data_store));
            }
            template <class Arg, class DataStore, std::enable_if_t<!has_key<KUpperBoundsMap, Arg>::value, int> = 0>
            void set_k_upper_bounds(Arg, DataStore const &data_store) {}

            template <class Arg, std::enable_if_t<has_key<KLowerBoundsMap, Arg>::value, int> = 0>
            GT_FUNCTION_DEVICE bool left_validate_k_pos(int_t pos) const {
                return pos >= device::at_key<Arg>(m_k_lower_bounds_map);
            }
            template <class Arg, std::enable_if_t<!has_key<KLowerBoundsMap, Arg>::value, int> = 0>
            GT_FUNCTION_DEVICE bool left_validate_k_pos(int_t pos) const {
                return true;
            }

            template <class Arg, std::enable_if_t<has_key<KUpperBoundsMap, Arg>::value, int> = 0>
            GT_FUNCTION_DEVICE bool right_validate_k_pos(int_t pos) const {
                return pos < device::at_key<Arg>(m_k_upper_bounds_map);
            }
            template <class Arg, std::enable_if_t<!has_key<KUpperBoundsMap, Arg>::value, int> = 0>
            GT_FUNCTION_DEVICE bool right_validate_k_pos(int_t pos) const {
                return true;
            }

          public:
            using max_extent_for_tmp_t = MaxExtentForTmp;
            using cache_sequence_t = CacheSequence;

            using ptr_holder_t = sid::ptr_holder_type<Composite>;
            using ptr_t = sid::ptr_type<Composite>;
            using strides_t = sid::strides_type<Composite>;
            using ptr_diff_t = sid::ptr_diff_type<Composite>;

            template <class Arg, class DataStore, std::enable_if_t<has_key<Composite, Arg>::value, int> = 0>
            void set_data_store(Arg, DataStore &data_store) {
                GT_STATIC_ASSERT(is_sid<DataStore>::value, "");

                at_key<Arg>(m_ptr_holder) = sid::get_origin(data_store);
                using stride_dims_t = get_keys<sid::strides_type<DataStore>>;
                auto const &src_strides = sid::get_strides(data_store);
                for_each_type<stride_dims_t>(local_domain_impl_::set_stride<Arg>(src_strides, m_strides));

                set_k_lower_bounds(Arg{}, data_store);
                set_k_upper_bounds(Arg{}, data_store);
            }

            template <class Arg, class DataStore, std::enable_if_t<!has_key<Composite, Arg>::value, int> = 0>
            void set_data_store(Arg, DataStore &) {}

            template <class Arg>
            GT_FUNCTION_DEVICE bool validate_k_pos(int_t pos) const {
                return left_validate_k_pos<Arg>(pos) && right_validate_k_pos<Arg>(pos);
            }

            ptr_holder_t m_ptr_holder;
            strides_t m_strides;
        };
        template <class Composite,
            class MaxExtentForTmp,
            class CacheSequence,
            class KLowerBoundsMap,
            class KUpperBoundsMap>
        struct is_local_domain<
            cuda_local_domain<Composite, MaxExtentForTmp, CacheSequence, KLowerBoundsMap, KUpperBoundsMap>>
            : std::true_type {};

        template <class Mss, class MaxExtentForTmp, bool IsStateful>
        struct get_local_domain<backend::cuda, Mss, MaxExtentForTmp, IsStateful> {
            GT_STATIC_ASSERT(is_extent<MaxExtentForTmp>::value, GT_INTERNAL_ERROR);

            using all_esf_args_t = extract_placeholders_from_mss<Mss>;
            GT_STATIC_ASSERT((meta::all_of<is_plh, all_esf_args_t>::value), GT_INTERNAL_ERROR);

            using caches_t = typename Mss::cache_sequence_t;
            GT_STATIC_ASSERT((meta::all_of<is_cache, caches_t>::value), GT_INTERNAL_ERROR);

            using local_caches_t = meta::filter<is_local_cache, caches_t>;
            using non_local_caches_t = meta::filter<meta::not_<is_local_cache>::apply, caches_t>;
            using local_cached_args_t = meta::transform<cache_parameter, local_caches_t>;
            using non_local_cached_args_t = meta::transform<cache_parameter, non_local_caches_t>;

            template <class Arg>
            using is_arg_used =
                bool_constant<!is_tmp_arg<Arg>::value || !meta::st_contains<local_cached_args_t, Arg>::value>;
            using esf_args_t = meta::filter<is_arg_used, all_esf_args_t>;

            template <class Arg>
            using arg_needs_k_size = meta::st_contains<non_local_cached_args_t, Arg>;
            using ksize_esf_args_t = meta::filter<arg_needs_k_size, esf_args_t>;

            using k_lower_bounds_args_t = meta::filter<local_domain_impl_::has_k_lower_bound, ksize_esf_args_t>;

            using k_lower_bounds_map_t = hymap::from_keys_values<k_lower_bounds_args_t,
                meta::transform<local_domain_impl_::get_lower_bound, k_lower_bounds_args_t>>;

            using k_upper_bounds_map_t = hymap::from_keys_values<k_lower_bounds_args_t,
                meta::transform<local_domain_impl_::get_upper_bound, k_lower_bounds_args_t>>;

            using positionals_t = local_domain_impl_::positionals<IsStateful, !meta::is_empty<ksize_esf_args_t>::value>;

            using args_t = meta::concat<esf_args_t, positionals_t>;

            using composite_keys_t = meta::rename<sid::composite::keys, args_t>;

            using storages_t = meta::transform<local_domain_impl_::get_storage, args_t>;

            using composite_t = meta::rename<composite_keys_t::template values, storages_t>;

            using type =
                cuda_local_domain<composite_t, MaxExtentForTmp, caches_t, k_lower_bounds_map_t, k_upper_bounds_map_t>;
        };
    } // namespace local_domain_impl_

    template <class Composite,
        class MaxExtentForTmp,
        class CacheSequence,
        class KLowerBoundsMap,
        class KUpperBoundsMap,
        class TotalLengthMap>
    class local_domain {
        GT_STATIC_ASSERT(is_extent<MaxExtentForTmp>::value, GT_INTERNAL_ERROR);

        KLowerBoundsMap m_k_lower_bounds_map;
        KUpperBoundsMap m_k_upper_bounds_map;

        template <class Arg, class DataStore, std::enable_if_t<has_key<KLowerBoundsMap, Arg>::value, int> = 0>
        void set_k_lower_bounds(Arg, DataStore const &data_store) {
            at_key<Arg>(m_k_lower_bounds_map) = at_key<dim::k>(sid::get_lower_bounds(data_store));
        }
        template <class Arg, class DataStore, std::enable_if_t<!has_key<KLowerBoundsMap, Arg>::value, int> = 0>
        void set_k_lower_bounds(Arg, DataStore const &data_store) {}

        template <class Arg, class DataStore, std::enable_if_t<has_key<KUpperBoundsMap, Arg>::value, int> = 0>
        void set_k_upper_bounds(Arg, DataStore const &data_store) {
            at_key<Arg>(m_k_upper_bounds_map) = at_key<dim::k>(sid::get_upper_bounds(data_store));
        }
        template <class Arg, class DataStore, std::enable_if_t<!has_key<KUpperBoundsMap, Arg>::value, int> = 0>
        void set_k_upper_bounds(Arg, DataStore const &data_store) {}

        template <class Arg, std::enable_if_t<has_key<KLowerBoundsMap, Arg>::value, int> = 0>
        GT_FUNCTION_DEVICE bool left_validate_k_pos(int_t pos) const {
            return pos >= device::at_key<Arg>(m_k_lower_bounds_map);
        }
        template <class Arg, std::enable_if_t<!has_key<KLowerBoundsMap, Arg>::value, int> = 0>
        GT_FUNCTION_DEVICE bool left_validate_k_pos(int_t pos) const {
            return true;
        }

        template <class Arg, std::enable_if_t<has_key<KUpperBoundsMap, Arg>::value, int> = 0>
        GT_FUNCTION_DEVICE bool right_validate_k_pos(int_t pos) const {
            return pos < device::at_key<Arg>(m_k_upper_bounds_map);
        }
        template <class Arg, std::enable_if_t<!has_key<KUpperBoundsMap, Arg>::value, int> = 0>
        GT_FUNCTION_DEVICE bool right_validate_k_pos(int_t pos) const {
            return true;
        }

      public:
        using max_extent_for_tmp_t = MaxExtentForTmp;
        using cache_sequence_t = CacheSequence;

        using ptr_holder_t = sid::ptr_holder_type<Composite>;
        using ptr_t = sid::ptr_type<Composite>;
        using strides_t = sid::strides_type<Composite>;
        using ptr_diff_t = sid::ptr_diff_type<Composite>;

        template <class Arg, class DataStore, std::enable_if_t<has_key<Composite, Arg>::value, int> = 0>
        void set_data_store(Arg, DataStore &data_store) {
            GT_STATIC_ASSERT(is_sid<DataStore>::value, "");

            at_key<Arg>(m_ptr_holder) = sid::get_origin(data_store);
            using stride_dims_t = get_keys<sid::strides_type<DataStore>>;
            auto const &src_strides = sid::get_strides(data_store);
            for_each_type<stride_dims_t>(local_domain_impl_::set_stride<Arg>(src_strides, m_strides));

            at_key_with_default<typename DataStore::storage_info_t, local_domain_impl_::sink>(m_total_length_map) =
                data_store.info().padded_total_length();

            set_k_lower_bounds(Arg{}, data_store);
            set_k_upper_bounds(Arg{}, data_store);
        }

        template <class Arg, class DataStore, std::enable_if_t<!has_key<Composite, Arg>::value, int> = 0>
        void set_data_store(Arg, DataStore &) {}

        template <class Arg>
        GT_FUNCTION_DEVICE bool validate_k_pos(int_t pos) const {
            return left_validate_k_pos<Arg>(pos) && right_validate_k_pos<Arg>(pos);
        }

        ptr_holder_t m_ptr_holder;
        strides_t m_strides;

        TotalLengthMap m_total_length_map;
    };

    using local_domain_impl_::is_local_domain;

    template <class Backend, class Mss, class MaxExtentForTmp, bool IsStateful>
    using get_local_domain =
        typename local_domain_impl_::get_local_domain<Backend, Mss, MaxExtentForTmp, IsStateful>::type;
} // namespace gridtools

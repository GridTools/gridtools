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

#include "../common/array.hpp"
#include "../common/defs.hpp"
#include "../common/hymap.hpp"
#include "../meta.hpp"
#include "../storage/sid.hpp"
#include "arg.hpp"
#include "caches/cache_traits.hpp"
#include "dim.hpp"
#include "extent.hpp"
#include "positional.hpp"
#include "sid/composite.hpp"
#include "sid/concept.hpp"

namespace gridtools {
    namespace local_domain_impl_ {
        GT_META_LAZY_NAMESPACE {
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
        }
        GT_META_DELEGATE_TO_LAZY(get_storage, class Arg, Arg);
        GT_META_DELEGATE_TO_LAZY(positionals, (bool IsStateful, bool NeedK), (IsStateful, NeedK));
        GT_META_DELEGATE_TO_LAZY(add_positional, (class Args, bool IsStateful), (Args, IsStateful));

        template <class Arg>
        GT_META_DEFINE_ALIAS(get_storage_info, meta::id, (typename Arg::data_store_t::storage_info_t));
    } // namespace local_domain_impl_

    /**
     * This class extracts the proper iterators/storages from the full domain to adapt it for a particular functor.
     */
    template <class Backend, class EsfArgs, class MaxExtentForTmp, class CacheSequence, bool IsStateful>
    struct local_domain {
      private:
        GT_STATIC_ASSERT(is_extent<MaxExtentForTmp>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((meta::all_of<is_plh, EsfArgs>::value), GT_INTERNAL_ERROR);

        using local_caches_t = GT_META_CALL(meta::filter, (is_local_cache, CacheSequence));
        using non_local_caches_t = GT_META_CALL(meta::filter, (meta::not_<is_local_cache>::apply, CacheSequence));
        using local_cached_args_t = GT_META_CALL(meta::transform, (cache_parameter, local_caches_t));
        using non_local_cached_args_t = GT_META_CALL(meta::transform, (cache_parameter, non_local_caches_t));

        template <class Arg>
        GT_META_DEFINE_ALIAS(is_arg_used,
            bool_constant,
            (!is_tmp_arg<Arg>::value || !std::is_same<Backend, backend::cuda>::value ||
                !meta::st_contains<local_cached_args_t, Arg>::value));
        using used_esf_args_t = GT_META_CALL(meta::filter, (is_arg_used, EsfArgs));

        template <class Arg>
        GT_META_DEFINE_ALIAS(arg_needs_k_size,
            bool_constant,
            (std::is_same<Backend, backend::cuda>::value && meta::st_contains<non_local_cached_args_t, Arg>::value));
        using ksize_esf_args_t = GT_META_CALL(meta::filter, (arg_needs_k_size, EsfArgs));

        template <class Arg>
        GT_META_DEFINE_ALIAS(arg_needs_total_length,
            bool_constant,
            (std::is_same<Backend, backend::mc>::value && is_tmp_arg<Arg>::value));
        using total_length_esf_args_t = GT_META_CALL(meta::filter, (arg_needs_total_length, EsfArgs));

        using positionals_t = GT_META_CALL(
            local_domain_impl_::positionals, (IsStateful, !meta::is_empty<ksize_esf_args_t>::value));

      public:
        using esf_args_t = GT_META_CALL(meta::concat, (used_esf_args_t, positionals_t));

      private:
        using ksize_storage_infos_t = GT_META_CALL(
            meta::dedup, (GT_META_CALL(meta::transform, (local_domain_impl_::get_storage_info, ksize_esf_args_t))));
        using total_length_storage_infos_t = GT_META_CALL(meta::dedup,
            (GT_META_CALL(meta::transform, (local_domain_impl_::get_storage_info, total_length_esf_args_t))));

        using composite_keys_t = GT_META_CALL(meta::rename, (sid::composite::keys, esf_args_t));

        using ksize_map_t = GT_META_CALL(hymap::from_keys_values,
            (ksize_storage_infos_t, GT_META_CALL(meta::repeat, (meta::length<ksize_storage_infos_t>, int_t))));

        using total_length_map_t = GT_META_CALL(hymap::from_keys_values,
            (total_length_storage_infos_t,
                GT_META_CALL(meta::repeat, (meta::length<total_length_storage_infos_t>, uint_t))));

        using storages_t = GT_META_CALL(meta::transform, (local_domain_impl_::get_storage, esf_args_t));

        using composite_t = GT_META_CALL(meta::rename, (composite_keys_t::template values, storages_t));

      public:
        using type = local_domain;
        using max_extent_for_tmp_t = MaxExtentForTmp;
        using cache_sequence_t = CacheSequence;

        using ptr_holder_t = GT_META_CALL(sid::ptr_holder_type, composite_t);
        using ptr_t = GT_META_CALL(sid::ptr_type, composite_t);
        using strides_t = GT_META_CALL(sid::strides_type, composite_t);
        using ptr_diff_t = GT_META_CALL(sid::ptr_diff_type, composite_t);

        ptr_holder_t m_ptr_holder;
        strides_t m_strides;

        total_length_map_t m_total_length_map;
        ksize_map_t m_ksize_map;
    };

    template <class>
    struct is_local_domain : std::false_type {};

    template <class Backend, class EsfArgs, class MaxExtentForTmp, class CacheSequence, bool IsStateful>
    struct is_local_domain<local_domain<Backend, EsfArgs, MaxExtentForTmp, CacheSequence, IsStateful>>
        : std::true_type {};
} // namespace gridtools

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

            template <>
            struct get_storage<positional> {
                using type = positional;
            };

            template <class Args, bool IsStateful>
            struct get_args : meta::lazy::id<Args> {};

            template <class Args>
            struct get_args<Args, true> : meta::lazy::push_back<Args, positional> {};
        }
        GT_META_DELEGATE_TO_LAZY(get_storage, class Arg, Arg);
        GT_META_DELEGATE_TO_LAZY(get_args, (class Args, bool IsStateful), (Args, IsStateful));

        template <class Arg>
        GT_META_DEFINE_ALIAS(get_storage_info, meta::id, (typename Arg::data_store_t::storage_info_t));
    } // namespace local_domain_impl_

    /**
     * This class extracts the proper iterators/storages from the full domain to adapt it for a particular functor.
     */
    template <class EsfArgs, class MaxExtentForTmp, class CacheSequence, bool IsStateful>
    struct local_domain {
        GT_STATIC_ASSERT(is_extent<MaxExtentForTmp>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((meta::all_of<is_plh, EsfArgs>::value), GT_INTERNAL_ERROR);

        using type = local_domain;

        using esf_args_t = GT_META_CALL(local_domain_impl_::get_args, (EsfArgs, IsStateful));
        using max_extent_for_tmp_t = MaxExtentForTmp;
        using cache_sequence_t = CacheSequence;

      private:
        using storage_infos_t = GT_META_CALL(
            meta::dedup, (GT_META_CALL(meta::transform, (local_domain_impl_::get_storage_info, EsfArgs))));

#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ == 9 && __CUDA_VER_MINOR__ < 2
        struct lazy_storage_info_keys_t : meta::lazy::rename<hymap::keys, storage_infos_t> {};
        using storage_info_t = typename lazy_storage_info_keys_t::type;
        struct lazy_compoisite_keys_t : meta::lazy::rename<sid::composite::keys, esf_args_t> {};
        using compoisite_keys_t = typename lazy_compoisite_keys_t::type;
#else
        using storage_info_keys_t = GT_META_CALL(meta::rename, (hymap::keys, storage_infos_t));
        using compoisite_keys_t = GT_META_CALL(meta::rename, (sid::composite::keys, esf_args_t));
#endif
        using total_length_map_t = GT_META_CALL(meta::rename,
            (storage_info_keys_t::template values,
                GT_META_CALL(meta::repeat, (meta::length<storage_info_keys_t>, uint_t))));

        using storages_t = GT_META_CALL(meta::transform, (local_domain_impl_::get_storage, esf_args_t));

        using composite_t = GT_META_CALL(meta::rename, (compoisite_keys_t::template values, storages_t));
        GT_STATIC_ASSERT(is_sid<composite_t>::value, GT_INTERNAL_ERROR);

      public:
        using ptr_holder_t = GT_META_CALL(sid::ptr_holder_type, composite_t);
        using ptr_t = GT_META_CALL(sid::ptr_type, composite_t);
        using strides_t = GT_META_CALL(sid::strides_type, composite_t);
        using ptr_diff_t = GT_META_CALL(sid::ptr_diff_type, composite_t);

        ptr_holder_t m_ptr_holder;
        strides_t m_strides;
        total_length_map_t m_total_length_map;
    };

    template <class>
    struct is_local_domain : std::false_type {};

    template <class EsfArgs, class MaxExtentForTmp, class CacheSequence, bool IsStateful>
    struct is_local_domain<local_domain<EsfArgs, MaxExtentForTmp, CacheSequence, IsStateful>> : std::true_type {};
} // namespace gridtools

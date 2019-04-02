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
#include "sid/concept.hpp"

namespace gridtools {

    namespace local_domain_impl_ {
        template <class Arg>
        GT_META_DEFINE_ALIAS(get_storage_info, meta::id, typename Arg::data_store_t::storage_info_t);

        template <class Arg>
        GT_META_DEFINE_ALIAS(get_storage_info_pair,
            meta::list,
            (typename Arg::data_store_t, typename Arg::data_store_t::storage_info_t));

        template <class Item>
        GT_META_DEFINE_ALIAS(get_strides, sid::strides_type, GT_META_CALL(meta::second, Item));

        template <class Arg>
        GT_META_DEFINE_ALIAS(get_ptr_holder, sid::ptr_holder_type, typename Arg::data_store_t);

        template <class Arg>
        GT_META_DEFINE_ALIAS(get_ptr, sid::ptr_type, typename Arg::data_store_t);

        struct call_f {
            template <class PtrHolder>
            GT_DEVICE auto operator()(PtrHolder const &holder) const GT_AUTO_RETURN(holder());
        };
    } // namespace local_domain_impl_

    /**
     * This class extracts the proper iterators/storages from the full domain to adapt it for a particular functor.
     */
    template <class EsfArgs, class MaxExtentForTmp, bool IsStateful>
    struct local_domain {
        GT_STATIC_ASSERT(is_extent<MaxExtentForTmp>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((meta::all_of<is_plh, EsfArgs>::value), GT_INTERNAL_ERROR);

        using type = local_domain;

        using esf_args_t = EsfArgs;
        using max_extent_for_tmp_t = MaxExtentForTmp;

        using tmp_storage_infos_t = GT_META_CALL(meta::dedup,
            (GT_META_CALL(meta::transform,
                (local_domain_impl_::get_storage_info, GT_META_CALL(meta::filter, (is_tmp_arg, EsfArgs))))));

      private:
        using inversed_storage_info_map_t = GT_META_CALL(
            meta::mp_inverse, (GT_META_CALL(meta::transform, (local_domain_impl_::get_storage_info_pair, EsfArgs))));

      public:
        using storage_infos_t = GT_META_CALL(meta::transform, (meta::first, inversed_storage_info_map_t));

      private:
        using sid_strides_values_t = GT_META_CALL(
            meta::transform, (local_domain_impl_::get_strides, inversed_storage_info_map_t));

#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ == 9 && __CUDA_VER_MINOR__ < 2
        struct lazy_strides_keys_t : meta::lazy::rename<hymap::keys, storage_infos_t> {};
        using strides_keys_t = typename lazy_strides_keys_t::type;
#else
        using strides_keys_t = GT_META_CALL(meta::rename, (hymap::keys, storage_infos_t));
#endif

        using strides_map_t = GT_META_CALL(meta::rename, (strides_keys_t::template values, sid_strides_values_t));

        using total_length_map_t = GT_META_CALL(meta::rename,
            (strides_keys_t::template values, GT_META_CALL(meta::repeat, (meta::length<storage_infos_t>, uint_t))));

        using ptr_holders_t = GT_META_CALL(meta::transform, (local_domain_impl_::get_ptr_holder, EsfArgs));
        using ptrs_t = GT_META_CALL(meta::transform, (local_domain_impl_::get_ptr, EsfArgs));

        using arg_keys_t = GT_META_CALL(meta::rename, (hymap::keys, EsfArgs));

        using ptr_holder_map_t = GT_META_CALL(meta::rename, (arg_keys_t::template values, ptr_holders_t));

      public:
        ptr_holder_map_t m_ptr_holder_map;
        total_length_map_t m_total_length_map;
        strides_map_t m_strides_map;

        using ptr_map_t = GT_META_CALL(meta::rename, (arg_keys_t::template values, ptrs_t));

        GT_DEVICE ptr_map_t make_ptr_map() const {
            return tuple_util::device::transform(local_domain_impl_::call_f{}, m_ptr_holder_map);
        }
    };

    template <class>
    struct is_local_domain : std::false_type {};

    template <class EsfArgs, class MaxExtentForTmp, bool IsStateful>
    struct is_local_domain<local_domain<EsfArgs, MaxExtentForTmp, IsStateful>> : std::true_type {};

    template <class>
    struct local_domain_is_stateful;

    template <class EsfArgs, class MaxExtentForTmp, bool IsStateful>
    struct local_domain_is_stateful<local_domain<EsfArgs, MaxExtentForTmp, IsStateful>> : bool_constant<IsStateful> {};
} // namespace gridtools

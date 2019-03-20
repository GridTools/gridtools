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

#include <boost/fusion/include/as_map.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/pair.hpp>

#include "../common/array.hpp"
#include "../common/defs.hpp"
#include "../common/generic_metafunctions/for_each.hpp"
#include "../common/hymap.hpp"
#include "../meta.hpp"
#include "../storage/sid.hpp"
#include "arg.hpp"
#include "extent.hpp"
#include "sid/concept.hpp"

namespace gridtools {

    namespace local_domain_impl_ {
        template <class Arg>
        GT_META_DEFINE_ALIAS(
            get_data_ptrs_elem, meta::id, (boost::fusion::pair<Arg, typename Arg::data_store_t::data_t *>));

        template <class Arg>
        GT_META_DEFINE_ALIAS(get_storage_info, meta::id, typename Arg::data_store_t::storage_info_t);

        template <class Arg>
        GT_META_DEFINE_ALIAS(get_storage_info_pair,
            meta::list,
            (typename Arg::data_store_t, typename Arg::data_store_t::storage_info_t));

        template <class Item>
        GT_META_DEFINE_ALIAS(get_strides, sid::strides_type, GT_META_CALL(meta::second, Item));

        // compatibility block
        template <class T>
        GT_META_DEFINE_ALIAS(const_ptr, meta::id, T const *);
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
      public:
        using strides_map_t = GT_META_CALL(meta::rename, (strides_keys_t::template values, sid_strides_values_t));

      private:
        using arg_to_data_ptr_map_t = GT_META_CALL(meta::transform, (local_domain_impl_::get_data_ptrs_elem, EsfArgs));
        using data_ptr_fusion_map = typename boost::fusion::result_of::as_map<arg_to_data_ptr_map_t>::type;
        using size_array_t = array<uint_t, meta::length<storage_infos_t>::value>;

      public:
        // compatibility block.
        using storage_info_ptr_list = GT_META_CALL(meta::transform, (local_domain_impl_::const_ptr, storage_infos_t));
        using storage_info_ptr_fusion_list = typename boost::fusion::result_of::as_vector<storage_info_ptr_list>::type;

        data_ptr_fusion_map m_local_data_ptrs;
        size_array_t m_local_padded_total_lengths;
        // disabled: strides_map_t m_strides_map;
        storage_info_ptr_fusion_list m_local_storage_info_ptrs;

        struct strides_filler_f {
            storage_info_ptr_fusion_list const &m_storage_info_ptrs;
            strides_map_t &m_dst;

            template <class Index>
            GT_FUNCTION void operator()() const {
                auto *src = boost::fusion::at<Index>(m_storage_info_ptrs);
                using storage_info_t = GT_META_CALL(meta::at, (storage_infos_t, Index));
// HACK!!! Hopefully this code will gone soon.
#ifndef __CUDACC__
                // shortcut for non cuda backends
                assert(src);
                host_device::at_key<storage_info_t>(m_dst) =
                    storage_sid_impl_::convert_strides_f<typename storage_info_t::layout_t>{}(src->strides());
#else
                if (src)
                    host_device::at_key<storage_info_t>(m_dst) =
                        storage_sid_impl_::convert_strides_f<typename storage_info_t::layout_t>{}(src->strides());
                else
                    host_device::at_key<storage_info_t>(m_dst) = {};
#endif
            }
        };

        GT_FUNCTION void init_strides_map(strides_map_t &dst) const {
            host_device::for_each_type<GT_META_CALL(meta::make_indices_for, storage_infos_t)>(
                strides_filler_f{m_local_storage_info_ptrs, dst});
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

#ifdef GT_USE_GPU
#include "../common/cuda_util.hpp"

namespace gridtools {
    // Force cloning to cuda device, even though local_domain is not trivially copyable because of boost fusion
    // containers implementation.
    namespace cuda_util {
        template <class EsfArgs, class MaxExtentForTmp, bool IsStateful>
        struct is_cloneable<local_domain<EsfArgs, MaxExtentForTmp, IsStateful>> : std::true_type {};
    } // namespace cuda_util
} // namespace gridtools

#endif

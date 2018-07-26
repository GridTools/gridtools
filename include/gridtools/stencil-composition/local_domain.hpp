/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

#include <boost/fusion/include/as_map.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/pair.hpp>

#include "../common/cuda_util.hpp"
#include "../common/defs.hpp"
#include "../common/generic_metafunctions/meta.hpp"
#include "../common/generic_metafunctions/type_traits.hpp"

#include "./arg.hpp"
#include "./extent.hpp"

namespace gridtools {

    namespace _impl {
        namespace local_domain_details {
            template <class Arg, class DataStore = typename Arg::data_store_t>
            GT_META_DEFINE_ALIAS(get_data_ptrs_elem,
                meta::id,
                (boost::fusion::pair<Arg, array<typename DataStore::data_t *, DataStore::num_of_storages>>));

            template <class Arg>
            GT_META_DEFINE_ALIAS(get_storage_info, meta::id, typename Arg::data_store_t::storage_info_t);

            template <class Args>
            GT_META_DEFINE_ALIAS(
                get_storage_infos, meta::dedup, (GT_META_CALL(meta::transform, (get_storage_info, Args))));

            template <class T>
            GT_META_DEFINE_ALIAS(add_const_ptr, meta::id, add_pointer_t<add_const_t<T>>);

            template <class StorageInfo>
            GT_META_DEFINE_ALIAS(get_strides_elem,
                meta::id,
                (boost::fusion::pair<StorageInfo,
                    array<uint_t,
                        StorageInfo::layout_t::unmasked_length == 0 ? 0
                                                                    : StorageInfo::layout_t::unmasked_length - 1>>));

            template <class StorageInfo>
            GT_META_DEFINE_ALIAS(get_size_elem, meta::id, (boost::fusion::pair<StorageInfo, uint_t>));

        } // namespace local_domain_details
    }     // namespace _impl

    /**
     * This class extract the proper iterators/storages from the full domain
     * to adapt it for a particular functor. This version does not provide grid
     * to the function operator
     *
     * @tparam StoragePointers The mpl vector of the storage pointer types
     * @tparam MetaData The mpl vector of the meta data pointer types sequence
     * @tparam EsfArgs The mpl vector of the args (i.e. placeholders for the storages)
                       for the current ESF
     * @tparam IsStateful The flag stating if the local_domain is aware of the position in the iteration domain
     */
    template <class EsfArgs, class MaxExtentForTmp, bool IsStateful>
    struct local_domain {
        GRIDTOOLS_STATIC_ASSERT(is_extent<MaxExtentForTmp>::value, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((meta::all_of<is_arg, EsfArgs>::value), GT_INTERNAL_ERROR);

        using type = local_domain;

        using esf_args = EsfArgs;
        using max_extent_for_tmp_t = MaxExtentForTmp;

        using arg_to_data_ptr_map_t = GT_META_CALL(
            meta::transform, (_impl::local_domain_details::get_data_ptrs_elem, EsfArgs));

        using storage_info_list = GT_META_CALL(_impl::local_domain_details::get_storage_infos, EsfArgs);
        using tmp_storage_info_list = GT_META_CALL(
            _impl::local_domain_details::get_storage_infos, (GT_META_CALL(meta::filter, (is_tmp_arg, EsfArgs))));

        using storage_info_ptr_list = GT_META_CALL(
            meta::transform, (_impl::local_domain_details::add_const_ptr, storage_info_list));
        using tmp_storage_info_ptr_list = GT_META_CALL(
            meta::transform, (_impl::local_domain_details::add_const_ptr, tmp_storage_info_list));

        using storage_info_to_strides_map_t = GT_META_CALL(
            meta::transform, (_impl::local_domain_details::get_strides_elem, storage_info_list));
        using storage_info_to_size_map_t = GT_META_CALL(
            meta::transform, (_impl::local_domain_details::get_size_elem, storage_info_list));

        using data_ptr_fusion_map = typename boost::fusion::result_of::as_map<arg_to_data_ptr_map_t>::type;
        using strides_fusion_map = typename boost::fusion::result_of::as_map<storage_info_to_strides_map_t>::type;
        using size_fusion_map = typename boost::fusion::result_of::as_map<storage_info_to_size_map_t>::type;

        template <class N>
        struct get_arg : meta::lazy::at_c<EsfArgs, N::value> {};

        data_ptr_fusion_map m_local_data_ptrs;
        strides_fusion_map m_local_strides;
        size_fusion_map m_local_padded_total_lengths;
    };

    template <class>
    struct is_local_domain : std::false_type {};

    template <class EsfArgs, class MaxExtentForTmp, bool IsStateful>
    struct is_local_domain<local_domain<EsfArgs, MaxExtentForTmp, IsStateful>> : std::true_type {};

    template <class>
    struct local_domain_is_stateful;

    template <class EsfArgs, class MaxExtentForTmp, bool IsStateful>
    struct local_domain_is_stateful<local_domain<EsfArgs, MaxExtentForTmp, IsStateful>> : bool_constant<IsStateful> {};

    template <class>
    struct local_domain_esf_args;

    // Force cloning to cuda device, even though local_domain is not trivially copyable because of boost fusion
    // containers implementation.
    namespace cuda_util {
        template <class EsfArgs, class MaxExtentForTmp, bool IsStateful>
        struct is_cloneable<local_domain<EsfArgs, MaxExtentForTmp, IsStateful>> : std::true_type {};
    } // namespace cuda_util

} // namespace gridtools

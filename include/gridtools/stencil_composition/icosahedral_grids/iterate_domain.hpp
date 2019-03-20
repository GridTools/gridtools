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
#include <boost/fusion/include/at_key.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/mpl/size.hpp>
#include <type_traits>

#include "../../common/array.hpp"
#include "../../common/defs.hpp"
#include "../../common/gt_assert.hpp"
#include "../../meta.hpp"
#include "../caches/cache_metafunctions.hpp"
#include "../iterate_domain_aux.hpp"
#include "../local_domain.hpp"
#include "../pos3.hpp"
#include "../sid/multi_shift.hpp"

namespace gridtools {
    /**
       This class is basically the iterate domain. It contains the
       ways to access data and the implementation of iterating on neighbors.
     */
    template <typename IterateDomainImpl, typename IterateDomainArguments>
    class iterate_domain {
        using local_domain_t = typename IterateDomainArguments::local_domain_t;
        using backend_ids_t = typename IterateDomainArguments::backend_ids_t;
        using storage_info_ptrs_t = typename local_domain_t::storage_info_ptr_fusion_list;
        using ij_cache_args_t = GT_META_CALL(ij_cache_args, typename IterateDomainArguments::cache_sequence_t);

        // the number of different storage metadatas used in the current functor
        static const uint_t n_meta_storages = boost::mpl::size<storage_info_ptrs_t>::value;

        GT_STATIC_ASSERT(is_local_domain<local_domain_t>::value, GT_INTERNAL_ERROR);

      protected:
        using strides_cached_t = typename local_domain_t::strides_map_t;

      private:
        using array_index_t = array<int_t, n_meta_storages>;

        local_domain_t const &m_local_domain;
        array_index_t m_index;

        /**
           @brief returns the strides as const reference
        */
        GT_FUNCTION strides_cached_t const &strides() const {
            return static_cast<IterateDomainImpl const *>(this)->strides_impl();
        }

        /**
           @brief returns the strides as const reference
        */
        GT_FUNCTION strides_cached_t &GT_RESTRICT strides() {
            return static_cast<IterateDomainImpl *>(this)->strides_impl();
        }

        template <class Dim, class Offset>
        GT_FUNCTION void increment(Offset const &offset) {
            do_increment<Dim, local_domain_t>(offset, strides(), m_index);
        }

      protected:
        /**@brief constructor of the iterate_domain struct

           It assigns the storage pointers to the first elements of
           the data fields (for all the data_fields present in the
           current evaluation), and the indexes to access the data
           fields (one index per storage instance, so that one index
           might be shared among several data fields)
        */
        GT_FUNCTION iterate_domain(local_domain_t const &local_domain_) : m_local_domain(local_domain_) {}

      public:
        static constexpr bool has_k_caches = false;

        /**
           @brief recursively assignes all the strides

           copies them from the
           local_domain.m_local_metadata vector, and stores them into an instance of the
           gridtools::strides_cached class.
         */
        GT_FUNCTION void assign_stride_pointers() { strides() = m_local_domain.m_strides_map; }

        /**@brief method for initializing the index */
        GT_FUNCTION void initialize(pos3<uint_t> begin, pos3<uint_t> block_no, pos3<int_t> pos_in_block) {
            host_device::for_each_type<typename local_domain_t::storage_infos_t>(
                initialize_index<backend_ids_t, local_domain_t>(strides(), begin, block_no, pos_in_block, m_index));
        }

        template <class Offset = integral_constant<int_t, 1>>
        GT_FUNCTION void increment_i(Offset const &offset = {}) {
            increment<dim::i>(offset);
        }
        template <class Offset = integral_constant<int_t, 1>>
        GT_FUNCTION void increment_j(Offset const &offset = {}) {
            increment<dim::j>(offset);
        }
        template <class Offset = integral_constant<int_t, 1>>
        GT_FUNCTION void increment_k(Offset const &offset = {}) {
            increment<dim::k>(offset);
        }

        template <class Offset = integral_constant<int_t, 1>>
        GT_FUNCTION void increment_c(Offset const &offset = {}) {
            increment<dim::c>(offset);
        }

        GT_FUNCTION array_index_t const &index() const { return m_index; }

        GT_FUNCTION void set_index(array_index_t const &index) { m_index = index; }

        template <class Arg,
            intent Intent,
            uint_t Color,
            class Accessor,
            class Res = typename deref_type<Arg, Intent>::type,
            enable_if_t<meta::st_contains<ij_cache_args_t, Arg>::value, int> = 0>
        GT_FUNCTION Res deref(Accessor const &acc) const {
            return static_cast<IterateDomainImpl const *>(this)->template get_ij_cache_value<Arg, Color, Res>(acc);
        }

        template <class Arg,
            intent Intent,
            uint_t Color,
            class Accessor,
            class Res = typename deref_type<Arg, Intent>::type,
            enable_if_t<!meta::st_contains<ij_cache_args_t, Arg>::value, int> = 0>
        GT_FUNCTION Res deref(Accessor const &acc) const {
            using storage_info_t = typename Arg::data_store_t::storage_info_t;

            typedef typename Arg::data_store_t::storage_info_t storage_info_t;
            typedef typename Arg::data_store_t::data_t data_t;

            // this index here describes the position of the storage info in the m_index array (can be different to the
            // storage info id)
            static constexpr auto storage_info_index =
                meta::st_position<typename local_domain_t::storage_info_ptr_list, storage_info_t const *>::value;

            auto pointer_offset = m_index[storage_info_index];
            sid::multi_shift(pointer_offset, host_device::at_key<storage_info_t>(strides()), acc);

            assert(pointer_oob_check<storage_info_t>(m_local_domain, pointer_offset));

            conditional_t<Intent == intent::in, data_t const, data_t> *ptr =
                boost::fusion::at_key<Arg>(m_local_domain.m_local_data_ptrs) + pointer_offset;

            return IterateDomainImpl::template deref_impl<Arg>(ptr);
        }
    };
} // namespace gridtools

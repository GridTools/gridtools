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

#include "../../common/array.hpp"
#include "../../common/defs.hpp"
#include "../../common/gt_assert.hpp"
#include "../../common/hymap.hpp"
#include "../../meta.hpp"
#include "../caches/cache_metafunctions.hpp"
#include "../iterate_domain_aux.hpp"
#include "../local_domain.hpp"
#include "../pos3.hpp"
#include "../sid/multi_shift.hpp"
#include "dim.hpp"

namespace gridtools {
    /**
       This class is basically the iterate domain. It contains the
       ways to access data and the implementation of iterating on neighbors.
     */
    template <typename IterateDomainImpl, typename IterateDomainArguments>
    class iterate_domain {
        using local_domain_t = typename IterateDomainArguments::local_domain_t;
        using backend_t = typename IterateDomainArguments::backend_t;
        using ij_cache_args_t = GT_META_CALL(ij_cache_args, typename IterateDomainArguments::cache_sequence_t);

        // the number of different storage metadatas used in the current functor
        static const uint_t n_meta_storages = meta::length<typename local_domain_t::strides_kinds_t>::value;

        GT_STATIC_ASSERT(is_local_domain<local_domain_t>::value, GT_INTERNAL_ERROR);

        using array_index_t = array<int_t, n_meta_storages>;

        local_domain_t const &m_local_domain;
        array_index_t m_index;

        template <class Dim, class Offset>
        GT_FUNCTION void increment(Offset offset) {
            do_increment<Dim, local_domain_t>(offset, m_local_domain.m_strides_map, m_index);
        }

      protected:
        /**@brief constructor of the iterate_domain struct

           It assigns the storage pointers to the first elements of
           the data fields (for all the data_fields present in the
           current evaluation), and the indexes to access the data
           fields (one index per storage instance, so that one index
           might be shared among several data fields)
        */
        GT_FUNCTION_DEVICE iterate_domain(local_domain_t const &local_domain_) : m_local_domain(local_domain_) {}

      public:
        static constexpr bool has_k_caches = false;

        /**@brief method for initializing the index */
        GT_FUNCTION void initialize(pos3<uint_t> begin, pos3<uint_t> block_no, pos3<int_t> pos_in_block) {
            host_device::for_each_type<typename local_domain_t::strides_kinds_t>(
                initialize_index<backend_t, local_domain_t>(
                    m_local_domain.m_strides_map, begin, block_no, pos_in_block, m_index));
        }

        template <class Offset = integral_constant<int_t, 1>>
        GT_FUNCTION void increment_i(Offset offset = {}) {
            increment<dim::i>(offset);
        }
        template <class Offset = integral_constant<int_t, 1>>
        GT_FUNCTION void increment_c(Offset offset = {}) {
            increment<dim::c>(offset);
        }
        template <class Offset = integral_constant<int_t, 1>>
        GT_FUNCTION void increment_j(Offset offset = {}) {
            increment<dim::j>(offset);
        }
        template <class Offset = integral_constant<int_t, 1>>
        GT_FUNCTION void increment_k(Offset offset = {}) {
            increment<dim::k>(offset);
        }

        GT_FUNCTION array_index_t const &index() const { return m_index; }

        GT_FUNCTION void set_index(array_index_t const &index) { m_index = index; }

        template <class Arg,
            intent Intent,
            uint_t Color,
            class Accessor,
            class Res = typename deref_type<Arg, Intent>::type,
            enable_if_t<meta::st_contains<ij_cache_args_t, Arg>::value, int> = 0>
        GT_FUNCTION Res deref(Accessor acc) const {
            return static_cast<IterateDomainImpl const *>(this)->template get_ij_cache_value<Arg, Color, Res>(
                std::move(acc));
        }

        template <class Arg,
            intent Intent,
            uint_t Color,
            class Accessor,
            class Res = typename deref_type<Arg, Intent>::type,
            enable_if_t<!meta::st_contains<ij_cache_args_t, Arg>::value, int> = 0>
        GT_FUNCTION Res deref(Accessor acc) const {
            using storage_info_t = typename Arg::data_store_t::storage_info_t;

            typedef typename Arg::data_store_t::storage_info_t storage_info_t;
            typedef typename Arg::data_store_t::data_t data_t;

            // this index here describes the position of the storage info in the m_index array (can be different to the
            // storage info id)
            static constexpr auto storage_info_index =
                meta::st_position<typename local_domain_t::strides_kinds_t, storage_info_t>::value;

            int_t pointer_offset = m_index[storage_info_index];
            sid::multi_shift(
                pointer_offset, host_device::at_key<storage_info_t>(m_local_domain.m_strides_map), std::move(acc));

            assert(pointer_oob_check<storage_info_t>(m_local_domain, pointer_offset));

            conditional_t<Intent == intent::in, data_t const, data_t> *ptr =
                gridtools::host_device::at_key<Arg>(m_local_domain.m_ptr_holder_map)() + pointer_offset;

            return IterateDomainImpl::template deref_impl<Arg>(ptr);
        }
    };
} // namespace gridtools

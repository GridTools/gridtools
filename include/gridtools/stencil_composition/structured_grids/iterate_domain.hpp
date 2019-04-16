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

#include <boost/fusion/adapted/std_tuple.hpp>
#include <boost/fusion/include/at.hpp>

#include "../../common/defs.hpp"
#include "../../common/gt_assert.hpp"
#include "../../common/host_device.hpp"
#include "../../common/hymap.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../caches/cache_metafunctions.hpp"
#include "../iterate_domain_aux.hpp"
#include "../local_domain.hpp"
#include "../pos3.hpp"
#include "../sid/concept.hpp"
#include "../sid/multi_shift.hpp"
#include "dim.hpp"

namespace gridtools {
    /**@brief class managing the memory accesses, indices increment

       This class gets instantiated in the backend-specific code, and has a different implementation for
       each backend (see CRTP pattern). It is instantiated within the kernel (e.g. in the device code),
       and drives all the operations which are performed at the innermost level. In particular
       the computation/increment of the useful addresses in memory, given the iteration point,
       the storage placeholders/metadatas and their offsets.
     */
    template <typename IterateDomainImpl, class IterateDomainArguments>
    class iterate_domain {
      private:
        using local_domain_t = typename IterateDomainArguments::local_domain_t;
        GT_STATIC_ASSERT(is_local_domain<local_domain_t>::value, GT_INTERNAL_ERROR);

        using caches_t = typename IterateDomainArguments::cache_sequence_t;
        using ij_cache_args_t = GT_META_CALL(ij_cache_args, caches_t);
        using k_cache_args_t = GT_META_CALL(k_cache_args, caches_t);

        // the number of different storage metadatas used in the current functor
        static const uint_t n_meta_storages = meta::length<typename local_domain_t::strides_kinds_t>::value;

      protected:
        using iterate_domain_arguments_t = IterateDomainArguments;

        GT_FUNCTION_DEVICE iterate_domain(local_domain_t const &local_domain_)
            : local_domain(local_domain_), m_ptr_map(local_domain_.make_ptr_map()) {}

      public:
        using array_index_t = array<int_t, n_meta_storages>;

      private:
        // ******************* members *******************
        local_domain_t const &local_domain;
        typename local_domain_t::ptr_map_t m_ptr_map;
        array_index_t m_index;
        // ******************* end of members *******************

        template <class Dim, class Offset>
        GT_FUNCTION void increment(Offset const &offset) {
            do_increment<Dim, local_domain_t>(offset, local_domain.m_strides_map, m_index);
        }

      public:
        static constexpr bool has_k_caches = false;

        GT_FUNCTION array_index_t const &index() const { return m_index; }

        /**@brief method for setting the index array
         * This method is responsible of assigning the index for the memory access at
         * the location (i,j,k). Such index is shared among all the fields contained in the
         * same storage class instance, and it is not shared among different storage instances.
         */
        GT_FUNCTION void set_index(array_index_t const &index) { m_index = index; }

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

        /**@brief method for initializing the index */
        GT_FUNCTION void initialize(pos3<uint_t> begin, pos3<uint_t> block_no, pos3<int_t> pos_in_block) {
            using backend_t = typename IterateDomainArguments::backend_t;
            host_device::for_each_type<typename local_domain_t::strides_kinds_t>(
                initialize_index<backend_t, local_domain_t>(
                    local_domain.m_strides_map, begin, block_no, pos_in_block, m_index));
        }

        template <class Arg, class DataStore = typename Arg::data_store_t, class Data = typename DataStore::data_t>
        GT_FUNCTION Data *deref_for_k_cache(int_t k_offset) const {
            using storage_info_t = typename DataStore::storage_info_t;
            static constexpr auto storage_info_index =
                meta::st_position<typename local_domain_t::strides_kinds_t, storage_info_t>::value;

            auto offset = m_index[storage_info_index];
            sid::shift(offset,
                sid::get_stride<dim::k>(host_device::at_key<storage_info_t>(local_domain.m_strides_map)),
                k_offset);

            return pointer_oob_check<storage_info_t>(local_domain, offset)
                       ? gridtools::host_device::at_key<Arg>(m_ptr_map) + offset
                       : nullptr;
        }

        /** @brief method called in the apply methods of the functors.
         *
         * Specialization for the offset_tuple placeholder (i.e. for extended storages, containing multiple snapshots of
         * data fields with the same dimension and memory layout)
         */
        template <class Arg,
            intent Intent,
            class Accessor,
            class Res = typename deref_type<Arg, Intent>::type,
            enable_if_t<meta::st_contains<ij_cache_args_t, Arg>::value, int> = 0>
        GT_FUNCTION Res deref(Accessor acc) const {
            return static_cast<IterateDomainImpl const *>(this)->template get_ij_cache_value<Arg, Res>(std::move(acc));
        }

        template <class Arg,
            intent Intent,
            class Accessor,
            class Res = typename deref_type<Arg, Intent>::type,
            enable_if_t<meta::st_contains<k_cache_args_t, Arg>::value &&
                            !meta::st_contains<ij_cache_args_t, Arg>::value,
                int> = 0>
        GT_FUNCTION Res deref(Accessor acc) const {
            return static_cast<IterateDomainImpl const *>(this)->template get_k_cache_value<Arg, Res>(std::move(acc));
        }

        /**
         * @brief returns the value pointed by an accessor in case the value is a normal accessor (not global accessor
         * nor expression) and is not cached (i.e. is accessing main memory)
         */
        template <class Arg,
            intent Intent,
            class Accessor,
            class Res = typename deref_type<Arg, Intent>::type,
            enable_if_t<!meta::st_contains<ij_cache_args_t, Arg>::value &&
                            !meta::st_contains<k_cache_args_t, Arg>::value && is_accessor<Accessor>::value,
                int> = 0>
        GT_FUNCTION Res deref(Accessor accessor) const {
            using data_t = typename Arg::data_store_t::data_t;
            using storage_info_t = typename Arg::data_store_t::storage_info_t;

            static constexpr auto storage_info_index =
                meta::st_position<typename local_domain_t::strides_kinds_t, storage_info_t>::value;

            auto pointer_offset = m_index[storage_info_index];
            sid::multi_shift(
                pointer_offset, host_device::at_key<storage_info_t>(local_domain.m_strides_map), std::move(accessor));

            assert(pointer_oob_check<storage_info_t>(local_domain, pointer_offset));

            conditional_t<Intent == intent::in, data_t const, data_t> *ptr =
                gridtools::host_device::at_key<Arg>(m_ptr_map) + pointer_offset;

            return IterateDomainImpl::template deref_impl<Arg>(ptr);
        }
    };
} // namespace gridtools

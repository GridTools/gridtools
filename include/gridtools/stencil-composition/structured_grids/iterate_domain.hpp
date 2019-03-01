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

#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/at_key.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/mpl/size.hpp>

#include "../../common/defs.hpp"
#include "../../common/gt_assert.hpp"
#include "../../common/host_device.hpp"
#include "../../meta.hpp"
#include "../caches/cache_metafunctions.hpp"
#include "../global_accessor.hpp"
#include "../iterate_domain_aux.hpp"
#include "../local_domain.hpp"
#include "../pos3.hpp"

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
        static const uint_t n_meta_storages = meta::length<typename local_domain_t::storage_infos_t>::value;

      protected:
        using strides_cached_t =
            strides_cached<n_meta_storages - 1, typename local_domain_t::storage_info_ptr_fusion_list>;
        using iterate_domain_arguments_t = IterateDomainArguments;

        GT_FUNCTION iterate_domain(local_domain_t const &local_domain_) : local_domain(local_domain_) {}

      public:
        using array_index_t = array<int_t, n_meta_storages>;

      private:
        // ******************* members *******************
        local_domain_t const &local_domain;
        array_index_t m_index;
        // ******************* end of members *******************

        /**
           @brief returns the strides as const reference
        */
        GT_FUNCTION strides_cached_t const &strides() const {
            return static_cast<const IterateDomainImpl *>(this)->strides_impl();
        }

        /**
           @brief returns the strides
        */
        GT_FUNCTION strides_cached_t &strides() { return static_cast<IterateDomainImpl *>(this)->strides_impl(); }

        template <uint_t Coordinate>
        GT_FUNCTION void increment(int_t step) {
            do_increment<Coordinate, local_domain_t>(step, strides(), m_index);
        }
        template <uint_t Coordinate, int_t Step>
        GT_FUNCTION void increment() {
            do_increment<Coordinate, Step, local_domain_t>(strides(), m_index);
        }

        /**
         * @brief helper function that given an input in_ and a tuple t_ calls in_.operator() with the elements of the
         * tuple as arguments.
         *
         * For example, if the tuple is an accessor containing the offsets 1,2,3, and the input is a storage st_,
         * this function returns st_(1,2,3).
         *
         * \param container_ the input class
         * \param tuple_ the tuple
         * */
        template <typename Container, typename Tuple, size_t... Ids>
        GT_FUNCTION auto static tuple_to_container(
            Container const &container_, Tuple const &tuple_, meta::index_sequence<Ids...>)
            GT_AUTO_RETURN(container_(boost::fusion::at_c<Ids>(tuple_)...));

      public:
        static constexpr bool has_k_caches = false;

        template <typename BackendType>
        GT_FUNCTION void assign_stride_pointers() {
            boost::fusion::for_each(local_domain.m_local_storage_info_ptrs,
                assign_strides<BackendType, strides_cached_t, local_domain_t>{strides()});
        }

        GT_FUNCTION array_index_t const &index() const { return m_index; }

        /**@brief method for setting the index array
         * This method is responsible of assigning the index for the memory access at
         * the location (i,j,k). Such index is shared among all the fields contained in the
         * same storage class instance, and it is not shared among different storage instances.
         */
        GT_FUNCTION void set_index(array_index_t const &index) { m_index = index; }

        template <int_t Step = 1>
        GT_FUNCTION void increment_i() {
            increment<0, Step>();
        }
        template <int_t Step = 1>
        GT_FUNCTION void increment_j() {
            increment<1, Step>();
        }
        template <int_t Step = 1>
        GT_FUNCTION void increment_k() {
            increment<2, Step>();
        }

        GT_FUNCTION void increment_i(int_t step) { increment<0>(step); }
        GT_FUNCTION void increment_j(int_t step) { increment<1>(step); }
        GT_FUNCTION void increment_k(int_t step) { increment<2>(step); }

        /**@brief method for initializing the index */
        GT_FUNCTION void initialize(pos3<uint_t> begin, pos3<uint_t> block_no, pos3<int_t> pos_in_block) {
            using backend_ids_t = typename IterateDomainArguments::backend_ids_t;
            host_device::for_each_type<typename local_domain_t::storage_infos_t>(
                initialize_index_f<strides_cached_t, local_domain_t, array_index_t, backend_ids_t>{
                    strides(), begin, block_no, pos_in_block, m_index});
        }

        template <class Arg, class DataStore = typename Arg::data_store_t, class Data = typename DataStore::data_t>
        GT_FUNCTION Data *deref_for_k_cache(int_t k_offset) const {
            using storage_info_t = typename DataStore::storage_info_t;
            static constexpr auto storage_info_index =
                meta::st_position<typename local_domain_t::storage_infos_t, storage_info_t>::value;

            int_t offset = m_index[storage_info_index] +
                           stride<storage_info_t, 2>(strides().template get<storage_info_index>()) * k_offset;

            return pointer_oob_check<storage_info_t>(local_domain, offset)
                       ? boost::fusion::at_key<Arg>(local_domain.m_local_data_ptrs) + offset
                       : nullptr;
        }

        /**
         * @brief Method called in the apply methods of the functors.
         * Specialization for the global accessors placeholders.
         */
        template <class Arg, intent Intent, uint_t I>
        GT_FUNCTION typename Arg::data_store_t::data_t deref(global_accessor<I> const &) const {
            return *boost::fusion::at_key<Arg>(local_domain.m_local_data_ptrs);
        }

        /**
         * @brief method called in the apply methods of the functors.
         * Specialization for the global accessors placeholders with arguments.
         */
        template <class Arg, intent Intent, class Acc, class... Args>
        GT_FUNCTION auto deref(global_accessor_with_arguments<Acc, Args...> const &acc) const
            GT_AUTO_RETURN(tuple_to_container(*boost::fusion::at_key<Arg>(local_domain.m_local_data_ptrs),
                acc.get_arguments(),
                meta::index_sequence_for<Args...>()));

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
        GT_FUNCTION Res deref(Accessor const &acc) const {
            return static_cast<IterateDomainImpl const *>(this)->template get_ij_cache_value<Arg, Res>(acc);
        }

        template <class Arg,
            intent Intent,
            class Accessor,
            class Res = typename deref_type<Arg, Intent>::type,
            enable_if_t<meta::st_contains<k_cache_args_t, Arg>::value &&
                            !meta::st_contains<ij_cache_args_t, Arg>::value,
                int> = 0>
        GT_FUNCTION Res deref(Accessor const &acc) const {
            return static_cast<IterateDomainImpl const *>(this)->template get_k_cache_value<Arg, Res>(acc);
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
                            !meta::st_contains<k_cache_args_t, Arg>::value && is_accessor<Accessor>::value &&
                            !is_global_accessor<Accessor>::value,
                int> = 0>
        GT_FUNCTION Res deref(Accessor const &accessor) const {
            using data_t = typename Arg::data_store_t::data_t;
            using storage_info_t = typename Arg::data_store_t::storage_info_t;

            GT_STATIC_ASSERT(Accessor::n_dimensions <= storage_info_t::layout_t::masked_length,
                "requested accessor index lower than zero. Check that when you define the accessor you specify the "
                "dimenisons which you actually access. e.g. suppose that a storage linked to the accessor ```in``` has "
                "5 dimensions, and thus can be called with in(Dimensions<5>(-1)). Calling in(Dimensions<6>(-1)) brings "
                "you here.");

            static constexpr auto storage_info_index =
                meta::st_position<typename local_domain_t::storage_infos_t, storage_info_t>::value;

            int_t pointer_offset =
                m_index[storage_info_index] +
                compute_offset<storage_info_t>(strides().template get<storage_info_index>(), accessor);

            assert(pointer_oob_check<storage_info_t>(local_domain, pointer_offset));

            conditional_t<Intent == intent::in, data_t const, data_t> *ptr =
                boost::fusion::at_key<Arg>(local_domain.m_local_data_ptrs) + pointer_offset;

            return IterateDomainImpl::template deref_impl<Arg>(ptr);
        }
    };
} // namespace gridtools

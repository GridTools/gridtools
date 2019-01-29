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
#include <boost/type_traits/remove_reference.hpp>
#include <type_traits>

#include "../../common/array.hpp"
#include "../../common/generic_metafunctions/variadic_to_vector.hpp"
#include "../../common/generic_metafunctions/variadic_typedef.hpp"
#include "../esf_metafunctions.hpp"
#include "../iterate_domain_aux.hpp"
#include "../iterate_domain_fwd.hpp"
#include "../location_type.hpp"
#include "on_neighbors.hpp"
#include "position_offset_type.hpp"

namespace gridtools {

    /**
       This class is basically the iterate domain. It contains the
       ways to access data and the implementation of iterating on neighbors.
     */
    template <typename IterateDomainImpl, typename IterateDomainArguments>
    struct iterate_domain {
        typedef IterateDomainArguments iterate_domain_arguments_t;
        typedef typename iterate_domain_arguments_t::local_domain_t local_domain_t;

        typedef typename iterate_domain_arguments_t::backend_ids_t backend_ids_t;
        typedef typename iterate_domain_arguments_t::esf_sequence_t esf_sequence_t;

        typedef typename local_domain_t::esf_args esf_args_t;

        typedef backend_traits_from_id<typename backend_ids_t::backend_id_t> backend_traits_t;
        typedef
            typename backend_traits_from_id<typename backend_ids_t::backend_id_t>::template select_iterate_domain_cache<
                iterate_domain_arguments_t>::type iterate_domain_cache_t;

        typedef typename iterate_domain_cache_t::ij_caches_map_t ij_caches_map_t;
        typedef typename iterate_domain_cache_t::all_caches_t all_caches_t;

        GRIDTOOLS_STATIC_ASSERT((is_local_domain<local_domain_t>::value), GT_INTERNAL_ERROR);

        typedef typename local_domain_t::storage_info_ptr_fusion_list storage_info_ptrs_t;
        typedef typename local_domain_t::data_ptr_fusion_map data_ptrs_map_t;

        // the number of different storage metadatas used in the current functor
        static const uint_t N_META_STORAGES = boost::mpl::size<storage_info_ptrs_t>::value;

        typedef strides_cached<N_META_STORAGES - 1, storage_info_ptrs_t> strides_cached_t;

        using array_index_t = array<int_t, N_META_STORAGES>;

      protected:
        local_domain_t const &m_local_domain;

      private:
        array_index_t m_index;

      public:
        /**@brief constructor of the iterate_domain struct

           It assigns the storage pointers to the first elements of
           the data fields (for all the data_fields present in the
           current evaluation), and the indexes to access the data
           fields (one index per storage instance, so that one index
           might be shared among several data fields)
        */
        GT_FUNCTION iterate_domain(local_domain_t const &local_domain_) : m_local_domain(local_domain_) {}

        /**
           @brief returns the strides as const reference
        */
        GT_FUNCTION
        strides_cached_t const &RESTRICT strides() const {
            return static_cast<IterateDomainImpl const *>(this)->strides_impl();
        }

        /**
           @brief returns the strides as const reference
        */
        GT_FUNCTION
        strides_cached_t &RESTRICT strides() { return static_cast<IterateDomainImpl *>(this)->strides_impl(); }

        /**
           @brief recursively assignes all the strides

           copies them from the
           local_domain.m_local_metadata vector, and stores them into an instance of the
           gridtools::strides_cached class.
         */
        template <typename BackendType, typename Strides>
        GT_FUNCTION void assign_stride_pointers() {
            boost::fusion::for_each(m_local_domain.m_local_storage_info_ptrs,
                assign_strides<BackendType, strides_cached_t, local_domain_t>(strides()));
        }

        /**@brief method for initializing the index */
        GT_FUNCTION void initialize(pos3<uint_t> begin, pos3<uint_t> block_no, pos3<int_t> pos_in_block) {
            using backend_ids_t = typename iterate_domain_arguments_t::backend_ids_t;
            boost::fusion::for_each(m_local_domain.m_local_storage_info_ptrs,
                initialize_index_f<strides_cached_t, local_domain_t, array_index_t, backend_ids_t>{
                    strides(), begin, block_no, pos_in_block, m_index});
        }

      private:
        template <uint_t Coordinate, int_t Step>
        GT_FUNCTION void increment() {
            do_increment<Coordinate, Step>(m_local_domain, strides(), m_index);
        }
        template <uint_t Coordinate>
        GT_FUNCTION void increment(int_t step) {
            do_increment<Coordinate>(step, m_local_domain, strides(), m_index);
        }

        template <class Arg>
        struct arg_is_cached
            : index_is_cached<meta::st_position<typename local_domain_t::esf_args, Arg>::value, all_caches_t> {};

      public:
        template <int_t Step = 1>
        GT_FUNCTION void increment_i() {
            increment<0, Step>();
        }
        template <int_t Step = 1>
        GT_FUNCTION void increment_c() {
            increment<1, Step>();
        }
        template <int_t Step = 1>
        GT_FUNCTION void increment_j() {
            increment<2, Step>();
        }
        template <int_t Step = 1>
        GT_FUNCTION void increment_k() {
            increment<3, Step>();
        }

        GT_FUNCTION void increment_i(int_t step) { increment<0>(step); }
        GT_FUNCTION void increment_c(int_t step) { increment<1>(step); }
        GT_FUNCTION void increment_j(int_t step) { increment<2>(step); }
        GT_FUNCTION void increment_k(int_t step) { increment<3>(step); }

        GT_FUNCTION
        array_index_t const &index() const { return m_index; }

        GT_FUNCTION void set_index(array_index_t const &index) { m_index = index; }

        template <class Arg,
            enumtype::intent Intent,
            uint_t Color,
            class Accessor,
            class Res = typename deref_type<Arg, Intent>::type,
            enable_if_t<arg_is_cached<Arg>::value, int> = 0>
        GT_FUNCTION Res deref(Accessor const &acc) const {
            return static_cast<IterateDomainImpl const *>(this)
                ->template get_cache_value_impl<meta::st_position<typename local_domain_t::esf_args, Arg>::value,
                    Color,
                    Res>(acc);
        }

        template <class Arg,
            enumtype::intent Intent,
            uint_t /*Color*/,
            class Accessor,
            class Res = typename deref_type<Arg, Intent>::type,
            enable_if_t<!arg_is_cached<Arg>::value, int> = 0>
        GT_FUNCTION Res deref(Accessor const &acc) const {
            using storage_info_t = typename Arg::data_store_t::storage_info_t;

            typedef typename Arg::data_store_t::storage_info_t storage_info_t;
            typedef typename Arg::data_store_t::data_t data_t;

            // this index here describes the position of the storage info in the m_index array (can be different to the
            // storage info id)
            static constexpr auto storage_info_index =
                meta::st_position<typename local_domain_t::storage_info_ptr_list, storage_info_t const *>::value;

            int_t pointer_offset = m_index[storage_info_index] +
                                   compute_offset<storage_info_t>(strides().template get<storage_info_index>(), acc);

            assert(pointer_oob_check<storage_info_t>(m_local_domain, pointer_offset));

            conditional_t<Intent == enumtype::in, data_t const, data_t> *ptr =
                boost::fusion::at_key<Arg>(m_local_domain.m_local_data_ptrs) + pointer_offset;

            return IterateDomainImpl::template deref_impl<Arg>(ptr);
        }
    };

} // namespace gridtools

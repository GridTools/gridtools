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
#include "../../common/explode_array.hpp"
#include "../../common/generic_metafunctions/gt_remove_qualifiers.hpp"
#include "../../common/generic_metafunctions/variadic_to_vector.hpp"
#include "../../common/generic_metafunctions/variadic_typedef.hpp"
#include "../../storage/data_field_view.hpp"
#include "../esf_metafunctions.hpp"
#include "../iterate_domain_aux.hpp"
#include "../iterate_domain_fwd.hpp"
#include "../location_type.hpp"
#include "accessor_metafunctions.hpp"
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
        typedef typename iterate_domain_arguments_t::grid_t::grid_topology_t grid_topology_t;
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
        // the number of storages  used in the current functor
        static const uint_t N_STORAGES = boost::mpl::size<data_ptrs_map_t>::value;

        typedef strides_cached<N_META_STORAGES - 1, storage_info_ptrs_t> strides_cached_t;

        using array_index_t = array<int_t, N_META_STORAGES>;

        /**
         * metafunction that computes the return type of all operator() of an accessor
         */
        template <typename Accessor>
        struct accessor_return_type {
            typedef typename accessor_return_type_impl<Accessor, iterate_domain_arguments_t>::type type;
        };

        template <typename T>
        struct map_return_type;

        template <typename MapF, typename LT, typename Arg0, typename... Args>
        struct map_return_type<map_function<MapF, LT, Arg0, Args...>> {
            GRIDTOOLS_STATIC_ASSERT((is_accessor<Arg0>::value), GT_INTERNAL_ERROR);
            typedef typename remove_restrict_reference<typename accessor_return_type<Arg0>::type>::type type;
        };

        typedef typename compute_readonly_args_indices<typename iterate_domain_arguments_t::esf_sequence_t>::type
            readonly_args_indices_t;

        /**
         * metafunction that determines if a given accessor is associated with an placeholder holding a data field
         */
        template <typename Accessor>
        struct accessor_holds_data_field {
            typedef typename aux::accessor_holds_data_field<Accessor, iterate_domain_arguments_t>::type type;
        };

        /**
         * metafunction that determines if a given accessor is associated with an arg that is cached
         */
        template <typename Accessor>
        struct cache_access_accessor {
            typedef typename accessor_is_cached<Accessor, all_caches_t>::type type;
        };

        /**
         * metafunction that determines if a given accessor is associated with an arg holding a
         * standard field (i.e. not a data field)
         * and the parameter refers to a storage in main memory (i.e. is not cached)
         */
        template <typename Accessor>
        struct mem_access_with_standard_accessor {
            typedef typename boost::mpl::and_<
                typename boost::mpl::and_<
                    typename boost::mpl::not_<typename accessor_is_cached<Accessor, all_caches_t>::type>::type,
                    typename boost::mpl::not_<typename accessor_holds_data_field<Accessor>::type>::type>::type,
                typename is_accessor<Accessor>::type>::type type;
        };

      protected:
        local_domain_t const &m_local_domain;

      private:
        grid_topology_t const &m_grid_topology;
        // TODOMEETING do we need m_index?
        array_index_t m_index;

      public:
        /**@brief constructor of the iterate_domain struct

           It assigns the storage pointers to the first elements of
           the data fields (for all the data_fields present in the
           current evaluation), and the indexes to access the data
           fields (one index per storage instance, so that one index
           might be shared among several data fields)
        */
        GT_FUNCTION
        iterate_domain(local_domain_t const &local_domain_, grid_topology_t const &grid_topology)
            : m_local_domain(local_domain_), m_grid_topology(grid_topology) {}

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
           \ref strides_cached class.
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

        /** @brief method returning the data pointer of an accessor
            specialization for the accessor placeholders for standard storages

            this method is enabled only if the current placeholder dimension does not exceed the number of space
           dimensions of the storage class.
            I.e., if we are dealing with storages, not with storage lists or data fields (see concepts page for
           definitions)
        */

        template <uint_t Color, typename Accessor>
        GT_FUNCTION typename boost::enable_if<typename cache_access_accessor<Accessor>::type,
            typename accessor_return_type<Accessor>::type>::type
        operator()(static_uint<Color>, Accessor const &accessor) const {
            GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "Using EVAL is only allowed for an accessor type");
            return static_cast<IterateDomainImpl const *>(this)
                ->template get_cache_value_impl<Color, typename accessor_return_type<Accessor>::type>(accessor);
        }

        template <uint_t Color, typename Accessor>
        GT_FUNCTION typename boost::disable_if<typename cache_access_accessor<Accessor>::type,
            typename accessor_return_type<Accessor>::type>::type
        operator()(static_uint<Color>, Accessor const &accessor_) const {
            return get_value(accessor_, aux::get_data_pointer(m_local_domain, accessor_));
        }

        /**@brief returns the value of the memory at the given address, plus the offset specified by the arg
           placeholder
           \param accessor accessor proxying the storage ID and the offsets
           \param storage_pointer pointer to the first element of the specific data field used
        */
        // TODO This should be merged with structured grids
        template <typename Accessor, typename StoragePointer>
        GT_FUNCTION typename accessor_return_type<Accessor>::type get_value(
            Accessor const &accessor, StoragePointer const &RESTRICT storage_pointer) const {
            // getting information about the storage
            typedef typename Accessor::index_t index_t;
            typedef typename local_domain_t::template get_arg<index_t>::type arg_t;
            typedef typename arg_t::data_store_t::storage_info_t storage_info_t;
            typedef typename arg_t::data_store_t::data_t data_t;

            // this index here describes the position of the storage info in the m_index array (can be different to the
            // storage info id)
            static constexpr auto storage_info_index =
                meta::st_position<typename local_domain_t::storage_info_ptr_list, storage_info_t const *>::value;

            GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "Using EVAL is only allowed for an accessor type");

            assert(storage_pointer);
            data_t *RESTRICT real_storage_pointer = static_cast<data_t *>(storage_pointer);
            assert(real_storage_pointer);

            // control your instincts: changing the following
            // int_t to uint_t will prevent GCC from vectorizing (compiler bug)
            const int_t pointer_offset =
                m_index[storage_info_index] +
                compute_offset<storage_info_t>(strides().template get<storage_info_index>(), accessor);

            assert(pointer_oob_check(
                boost::fusion::at_c<storage_info_index>(m_local_domain.m_local_storage_info_ptrs), pointer_offset));

            return static_cast<const IterateDomainImpl *>(this)
                ->template get_value_impl<typename accessor_return_type<Accessor>::type, Accessor>(
                    real_storage_pointer, pointer_offset);
        }

        template <typename Accessor, typename StorageType>
        GT_FUNCTION typename accessor_return_type<Accessor>::type get_raw_value(
            Accessor const &accessor, StorageType *RESTRICT storage_pointer, int_t offset) const {
            // getting information about the storage
            GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "Using EVAL is only allowed for an accessor type");

            typedef typename Accessor::index_t index_t;
            typedef typename local_domain_t::template get_arg<index_t>::type arg_t;
            typedef typename arg_t::data_store_t::storage_info_t storage_info_t;
            typedef typename arg_t::data_store_t::data_t data_t;

            data_t *RESTRICT real_storage_pointer =
                static_cast<data_t *>(boost::fusion::at<index_t>(m_local_domain.m_local_data_ptrs).second[0]);

            assert(pointer_oob_check(
                boost::fusion::at_c<
                    meta::st_position<typename local_domain_t::storage_info_ptr_list, storage_info_t const *>::value>(
                    m_local_domain.m_local_storage_info_ptrs),
                offset));

            return static_cast<const IterateDomainImpl *>(this)
                ->template get_value_impl<typename accessor_return_type<Accessor>::type, Accessor>(
                    real_storage_pointer, offset);
        }

        /**
         * It dereferences the value of an accessor given its 4d (i,c,j,k) position_offset
         */
        template <uint_t ID, enumtype::intent Intent, typename LocationType, typename Extent, ushort_t FieldDimensions>
        GT_FUNCTION typename std::remove_reference<
            typename accessor_return_type<accessor<ID, Intent, LocationType, Extent, FieldDimensions>>::type>::type
        _evaluate(accessor<ID, Intent, LocationType, Extent, FieldDimensions>,
            position_offset_type const &RESTRICT position_offset) const {
            using accessor_t = accessor<ID, Intent, LocationType, Extent, FieldDimensions>;
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor<accessor_t>::value), "Using EVAL is only allowed for an accessor type");

            // getting information about the storage
            typedef typename accessor_t::index_t index_t;
            typedef typename local_domain_t::template get_arg<index_t>::type arg_t;
            typedef typename arg_t::data_store_t::storage_info_t storage_info_t;

            // this index here describes the position of the storage info in the m_index array (can be different to the
            // storage info id)
            static constexpr auto storage_info_index =
                meta::st_position<typename local_domain_t::storage_info_ptr_list, storage_info_t const *>::value;

            using location_type_t = typename accessor_t::location_type;
            // control your instincts: changing the following
            // int_t to uint_t will prevent GCC from vectorizing (compiler bug)
            const int_t pointer_offset =
                m_index[storage_info_index] +
                compute_offset<storage_info_t>(strides().template get<storage_info_index>(), position_offset);

            return get_raw_value(
                accessor_t(), boost::fusion::at<index_t>(m_local_domain.m_local_data_ptrs).second[0], pointer_offset);
        }
    };

} // namespace gridtools

/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include <type_traits>
#include <boost/type_traits/remove_reference.hpp>
#include "common/generic_metafunctions/is_not_same.hpp"
#include "common/generic_metafunctions/apply_to_sequence.hpp"
#include "common/generic_metafunctions/vector_to_set.hpp"
#include "common/generic_metafunctions/variadic_to_vector.hpp"
#include "common/generic_metafunctions/variadic_typedef.hpp"
#include "common/array.hpp"
#include "../../common/explode_array.hpp"
#include "common/generic_metafunctions/remove_restrict_reference.hpp"
#include "../location_type.hpp"
#include "stencil-composition/iterate_domain_impl_metafunctions.hpp"
#include "stencil-composition/total_storages.hpp"
#include "stencil-composition/iterate_domain_aux.hpp"
#include "stencil-composition/icosahedral_grids/accessor_metafunctions.hpp"
#include "on_neighbors.hpp"
#include "../iterate_domain_fwd.hpp"

namespace gridtools {

    /**
       This class is basically the iterate domain. It contains the
       ways to access data and the implementation of iterating on neighbors.
     */
    // template <typename PlcVector, typename GridType, typename LocationType>
    template < typename IterateDomainImpl >
    struct iterate_domain {
        typedef iterate_domain< IterateDomainImpl > type;

        typedef typename iterate_domain_impl_arguments< IterateDomainImpl >::type iterate_domain_arguments_t;
        typedef typename iterate_domain_arguments_t::local_domain_t local_domain_t;

        typedef typename iterate_domain_arguments_t::processing_elements_block_size_t processing_elements_block_size_t;

        typedef typename iterate_domain_arguments_t::backend_ids_t backend_ids_t;

        typedef typename iterate_domain_arguments_t::grid_t::grid_topology_t grid_topology_t;
        typedef typename grid_topology_t::layout_map_t layout_map_t;
        typedef typename iterate_domain_arguments_t::esf_sequence_t esf_sequence_t;

        typedef typename local_domain_t::esf_args esf_args_t;

        typedef typename backend_traits_from_id< backend_ids_t::s_backend_id >::template select_iterate_domain_cache<
            iterate_domain_arguments_t >::type iterate_domain_cache_t;

        typedef typename iterate_domain_cache_t::ij_caches_map_t ij_caches_map_t;
        typedef typename iterate_domain_cache_t::all_caches_t all_caches_t;

        GRIDTOOLS_STATIC_ASSERT((is_local_domain< local_domain_t >::value), "Internal Error: wrong type");
        typedef typename boost::remove_pointer<
            typename boost::mpl::at_c< typename local_domain_t::mpl_storages, 0 >::type >::type::value_type value_type;

        typedef typename local_domain_t::storage_metadata_map metadata_map_t;
        typedef typename local_domain_t::actual_args_type actual_args_type;

        // the number of different storage metadatas used in the current functor
        static const uint_t N_META_STORAGES = boost::mpl::size< metadata_map_t >::value;
        // the number of storages  used in the current functor
        static const uint_t N_STORAGES = boost::mpl::size< actual_args_type >::value;
        // the total number of snapshot (one or several per storage)
        static const uint_t N_DATA_POINTERS = total_storages< actual_args_type,
            boost::mpl::size< typename local_domain_t::mpl_storages >::type::value >::value;

        typedef array< void * RESTRICT, N_DATA_POINTERS > data_pointer_array_t;
        typedef strides_cached< N_META_STORAGES - 1, typename local_domain_t::storage_metadata_vector_t >
            strides_cached_t;

        /**@brief local class instead of using the inline (cond)?a:b syntax, because in the latter both branches get
         * compiled (generating sometimes a compile-time overflow) */
        template < bool condition, typename LocalD, typename Accessor >
        struct current_storage;

        template < typename LocalD, typename Accessor >
        struct current_storage< true, LocalD, Accessor > {
            static const uint_t value = 0;
        };

        template < typename LocalD, typename Accessor >
        struct current_storage< false, LocalD, Accessor > {
            static const uint_t value =
                (total_storages< typename LocalD::local_args_type, Accessor::index_type::value >::value);
        };

        /**
         * metafunction that computes the return type of all operator() of an accessor
         */
        template < typename Accessor >
        struct accessor_return_type {
            typedef typename accessor_return_type_impl< Accessor, iterate_domain_arguments_t >::type type;
        };

        template < typename T >
        struct map_return_type;

        template < typename MapF, typename LT, typename Arg0, typename... Args >
        struct map_return_type< map_function< MapF, LT, Arg0, Args... > > {
            GRIDTOOLS_STATIC_ASSERT((is_accessor< Arg0 >::value), "Error");
            typedef typename remove_restrict_reference< typename accessor_return_type< Arg0 >::type >::type type;
        };

        typedef typename compute_readonly_args_indices< typename iterate_domain_arguments_t::esf_sequence_t >::type
            readonly_args_indices_t;

        /**
         * metafunction that determines if a given accessor is associated with an placeholder holding a data field
         */
        template < typename Accessor >
        struct accessor_holds_data_field {
            typedef typename aux::accessor_holds_data_field< Accessor, iterate_domain_arguments_t >::type type;
        };

        /**
         * metafunction that determines if a given accessor is associated with an arg that is cached
         */
        template < typename Accessor >
        struct cache_access_accessor {
            typedef typename accessor_is_cached< Accessor, all_caches_t >::type type;
        };

        /**
         * metafunction that determines if a given accessor is associated with an arg holding a
         * standard field (i.e. not a data field)
         * and the parameter refers to a storage in main memory (i.e. is not cached)
         */
        template < typename Accessor >
        struct mem_access_with_standard_accessor {
            typedef typename aux::mem_access_with_standard_accessor< Accessor,
                all_caches_t,
                iterate_domain_arguments_t >::type type;
        };

      private:
        local_domain_t const &m_local_domain;
        grid_topology_t const &m_grid_topology;
        typedef array< int_t, N_META_STORAGES > array_index_t;
        // TODOMEETING do we need m_index?
        array_index_t m_index;

        array< uint_t, 4 > m_grid_position;

      public:
        /**@brief constructor of the iterate_domain struct

           It assigns the storage pointers to the first elements of
           the data fields (for all the data_fields present in the
           current evaluation), and the indexes to access the data
           fields (one index per storage instance, so that one index
           might be shared among several data fileds)
        */
        GT_FUNCTION
        iterate_domain(local_domain_t const &local_domain_, grid_topology_t const &grid_topology)
            : m_local_domain(local_domain_), m_grid_topology(grid_topology) {}

        /**
           @brief returns the array of pointers to the raw data
        */
        GT_FUNCTION
        data_pointer_array_t const &RESTRICT data_pointer() const {
            return static_cast< IterateDomainImpl const * >(this)->data_pointer_impl();
        }

        /**
           @brief returns the array of pointers to the raw data
        */
        GT_FUNCTION
        data_pointer_array_t &RESTRICT data_pointer() {
            return static_cast< IterateDomainImpl * >(this)->data_pointer_impl();
        }

        /**
           @brief returns the strides as const reference
        */
        GT_FUNCTION
        strides_cached_t const &RESTRICT strides() const {
            return static_cast< IterateDomainImpl const * >(this)->strides_impl();
        }

        /**
           @brief returns the strides as const reference
        */
        GT_FUNCTION
        strides_cached_t &RESTRICT strides() { return static_cast< IterateDomainImpl * >(this)->strides_impl(); }

        /** This functon set the addresses of the data values  before the computation
            begins.

            The EU stands for ExecutionUnit (thich may be a thread or a group of
            threasd. There are potentially two ids, one over i and one over j, since
            our execution model is parallel on (i,j). Defaulted to 1.
        */
        template < typename BackendType >
        GT_FUNCTION void assign_storage_pointers() {
            const uint_t EU_id_i = BackendType::processing_element_i();
            const uint_t EU_id_j = BackendType::processing_element_j();

            boost::mpl::for_each< typename reversed_range< uint_t, 0, N_STORAGES >::type >(
                assign_storage_functor< BackendType,
                    data_pointer_array_t,
                    typename local_domain_t::local_args_type,
                    typename local_domain_t::local_metadata_type,
                    metadata_map_t,
                    processing_elements_block_size_t >(
                    data_pointer(), m_local_domain.m_local_args, m_local_domain.m_local_metadata, EU_id_i, EU_id_j));
        }

        /**
           @brief recursively assignes all the strides

           copies them from the
           local_domain.m_local_metadata vector, and stores them into an instance of the
           \ref strides_cached class.
         */
        template < typename BackendType, typename Strides >
        GT_FUNCTION void assign_stride_pointers() {
            GRIDTOOLS_STATIC_ASSERT((is_strides_cached< Strides >::value), "internal error type");
            boost::mpl::for_each< metadata_map_t >(assign_strides_functor< BackendType,
                Strides,
                typename boost::fusion::result_of::as_vector< typename local_domain_t::local_metadata_type >::type,
                processing_elements_block_size_t >(strides(), m_local_domain.m_local_metadata));
        }

        /**@brief method for initializing the index */
        template < ushort_t Coordinate >
        GT_FUNCTION void initialize(uint_t const initial_pos = 0, uint_t const block = 0) {
            boost::mpl::for_each< metadata_map_t >(initialize_index_functor< Coordinate,
                strides_cached_t,
                typename boost::fusion::result_of::as_vector< typename local_domain_t::local_metadata_type >::type,
                array_index_t >(
                strides(), boost::fusion::as_vector(m_local_domain.m_local_metadata), initial_pos, block, m_index));
            static_cast< IterateDomainImpl * >(this)->template initialize_impl< Coordinate >();

            m_grid_position[Coordinate] = initial_pos;
        }

        /**@brief method for incrementing by 1 the index when moving forward along the given direction
           \tparam Coordinate dimension being incremented
           \tparam Execution the policy for the increment (e.g. forward/backward)
         */
        template < ushort_t Coordinate, typename Steps >
        GT_FUNCTION void increment() {
            boost::mpl::for_each< metadata_map_t >(increment_index_functor< Coordinate,
                strides_cached_t,
                typename boost::fusion::result_of::as_vector< typename local_domain_t::local_metadata_type >::type,
                array_index_t >(
                boost::fusion::as_vector(m_local_domain.m_local_metadata), Steps::value, m_index, strides()));
            static_cast< IterateDomainImpl * >(this)->template increment_impl< Coordinate, Steps >();
            m_grid_position[Coordinate] += Steps::value;
        }

        /**@brief method for incrementing the index when moving forward along the given direction

           \param steps_ the increment
           \tparam Coordinate dimension being incremented
         */
        template < ushort_t Coordinate >
        GT_FUNCTION void increment(int_t steps_) {
            boost::mpl::for_each< metadata_map_t >(increment_index_functor< Coordinate,
                strides_cached_t,
                typename boost::fusion::result_of::as_vector< typename local_domain_t::local_metadata_type >::type,
                array_index_t >(boost::fusion::as_vector(m_local_domain.m_local_metadata), steps_, m_index, strides()));
            static_cast< IterateDomainImpl * >(this)->template increment_impl< Coordinate >(steps_);

            m_grid_position[Coordinate] += steps_;
        }

        /**@brief getter for the index array */
        // TODO simplify this using just loops
        GT_FUNCTION
        void get_index(array< int_t, N_META_STORAGES > &index) const {
            for (int_t i = 0; i < N_META_STORAGES; ++i) {
                index[i] = m_index[i];
            }
        }

        GT_FUNCTION
        array< uint_t, 4 > const &position() const { return m_grid_position; }

        /**@brief getter for the index array */
        GT_FUNCTION
        void get_position(array< uint_t, 4 > &position) const { position = m_grid_position; }

        /**@brief method for setting the index array */
        template < typename Value >
        GT_FUNCTION void set_index(array< Value, N_META_STORAGES > const &index) {
            for (int_t i = 0; i < N_META_STORAGES; ++i) {
                m_index[i] = index[i];
            }
        }

        GT_FUNCTION
        void set_index(int index) {
            for (int_t i = 0; i < N_META_STORAGES; ++i) {
                m_index[i] = index;
            }
        }

        GT_FUNCTION
        void set_position(array< uint_t, 4 > const &position) { m_grid_position = position; }

        /** @brief method returning the data pointer of an accessor
            specialization for the accessor placeholders for standard storages

            this method is enabled only if the current placeholder dimension does not exceed the number of space
           dimensions of the storage class.
            I.e., if we are dealing with storages, not with storage lists or data fields (see concepts page for
           definitions)
        */
        template < typename Accessor >
        GT_FUNCTION
            typename boost::disable_if< typename accessor_holds_data_field< Accessor >::type, void * RESTRICT >::type
            get_data_pointer(Accessor const &accessor) const {
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");
            return (data_pointer())
                [current_storage< (Accessor::index_type::value == 0), local_domain_t, typename Accessor::type >::value];
        }

        template < uint_t Color, typename Accessor >
        GT_FUNCTION typename boost::enable_if< typename cache_access_accessor< Accessor >::type,
            typename accessor_return_type< Accessor >::type >::type
        operator()(static_uint< Color >, Accessor const &accessor) const {
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");
            return static_cast< IterateDomainImpl const * >(this)
                ->template get_cache_value_impl< Color, typename accessor_return_type< Accessor >::type >(accessor);
        }

        template < uint_t Color, typename Accessor >
        GT_FUNCTION typename boost::enable_if< typename mem_access_with_standard_accessor< Accessor >::type,
            typename accessor_return_type< Accessor >::type >::type
        operator()(static_uint< Color >, Accessor const &accessor_) const {
            return get_value(accessor_,
                (data_pointer())[current_storage< (Accessor::index_type::value == 0),
                    local_domain_t,
                    typename Accessor::type >::value]);
        }

        /** @brief return a the value in gmem pointed to by an accessor
        */
        template < typename ReturnType, typename StoragePointer >
        GT_FUNCTION ReturnType get_gmem_value(StoragePointer RESTRICT &storage_pointer
            // control your instincts: changing the following
            // int_t to uint_t will prevent GCC from vectorizing (compiler bug)
            ,
            const int_t pointer_offset) const {
            return *(storage_pointer + pointer_offset);
        }

        /**@brief returns the value of the memory at the given address, plus the offset specified by the arg
           placeholder
           \param arg placeholder containing the storage ID and the offsets
           \param storage_pointer pointer to the first element of the specific data field used
        */
        // TODO This should be merged with structured grids
        template < typename Accessor, typename StoragePointer >
        GT_FUNCTION typename accessor_return_type< Accessor >::type get_value(
            Accessor const &accessor, StoragePointer const &RESTRICT storage_pointer) const {

            // getting information about the storage
            typedef typename Accessor::index_type index_t;

            auto const storage_ = boost::fusion::at< index_t >(m_local_domain.m_local_args);

            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");

            using storage_type = typename std::remove_reference< decltype(*storage_) >::type;
            typename storage_type::value_type *RESTRICT real_storage_pointer =
                static_cast< typename storage_type::value_type * >(storage_pointer);

            typedef typename get_storage_pointer_accessor< local_domain_t, Accessor >::type storage_pointer_t;

            // getting information about the metadata
            typedef typename boost::mpl::at< metadata_map_t, typename storage_type::storage_info_type >::type
                metadata_index_t;

            pointer< const typename storage_type::storage_info_type > const metadata_ =
                boost::fusion::at< metadata_index_t >(m_local_domain.m_local_metadata);
            // getting the value
            // the following assert fails when an out of bound access is observed, i.e. either one of
            // i+offset_i or j+offset_j or k+offset_k is too large.
            // Most probably this is due to you specifying a positive offset which is larger than expected,
            // or maybe you did a mistake when specifying the extents in the placehoders definition
            assert((int)metadata_->size() > (m_index[metadata_index_t::value]));

            // the following assert fails when an out of bound access is observed,
            // i.e. when some offset is negative and either one of
            // i+offset_i or j+offset_j or k+offset_k is too small.
            // Most probably this is due to you specifying a negative offset which is
            // smaller than expected, or maybe you did a mistake when specifying the extents
            // in the placehoders definition.
            // If you are running a parallel simulation another common reason for this to happen is
            // the definition of an halo region which is too small in one direction
            // std::cout<<"Storage Index: "<<Accessor::index_type::value<<" + "<<(boost::fusion::at<typename
            // Accessor::index_type>(local_domain.local_args))->_index(arg.template
            // n<Accessor::n_dim>())<<std::endl;
            assert((int_t)(metadata_->index(m_grid_position)) >= 0);

            const int_t pointer_offset =
                (m_index[metadata_index_t::value]) +
                metadata_->_index(strides().template get< metadata_index_t::value >(), accessor.offsets());

            assert((int)metadata_->size() > pointer_offset);
            return static_cast< const IterateDomainImpl * >(this)
                ->template get_value_impl<
                    typename iterate_domain< IterateDomainImpl >::template accessor_return_type< Accessor >::type,
                    Accessor,
                    storage_pointer_t >(real_storage_pointer, pointer_offset);
        }

        template < typename Accessor, typename StoragePointer >
        GT_FUNCTION typename accessor_return_type< Accessor >::type get_raw_value(
            Accessor const &accessor, StoragePointer &RESTRICT storage_pointer, const uint_t offset) const {

            // getting information about the storage
            typedef typename Accessor::index_type index_t;

            typedef typename get_storage_pointer_accessor< local_domain_t, Accessor >::type storage_pointer_t;

            auto const storage_ = boost::fusion::at< index_t >(m_local_domain.m_local_args);

            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");

            using storage_type = typename std::remove_reference< decltype(*storage_) >::type;
            typename storage_type::value_type *RESTRICT real_storage_pointer =
                static_cast< typename storage_type::value_type * >(storage_pointer);

#ifndef NDEBUG
            typedef typename boost::mpl::at< metadata_map_t, typename storage_type::storage_info_type >::type
                metadata_index_t;

            pointer< const typename storage_type::storage_info_type > const metadata_ =
                boost::fusion::at< metadata_index_t >(m_local_domain.m_local_metadata);

            assert((int)metadata_->size() > offset);
#endif
            return static_cast< const IterateDomainImpl * >(this)
                ->template get_value_impl<
                    typename iterate_domain< IterateDomainImpl >::template accessor_return_type< Accessor >::type,
                    Accessor,
                    storage_pointer_t >(real_storage_pointer, offset);
        }

        /**
         * It dereferences the value of an accessor given its 4d (i,c,j,k) position_offset
         */
        template < uint_t ID,
            enumtype::intend Intend,
            typename LocationType,
            typename Extent,
            ushort_t FieldDimensions >
        GT_FUNCTION typename std::remove_reference<
            typename accessor_return_type< accessor< ID, Intend, LocationType, Extent, FieldDimensions > >::type >::type
            _evaluate(accessor< ID, Intend, LocationType, Extent, FieldDimensions >,
                array< int_t, 4 > const &RESTRICT position_offset) const {
            GRIDTOOLS_STATIC_ASSERT((LocationType::value == location_type_t::value), "error");

            using accessor_t = accessor< ID, Intend, LocationType, Extent, FieldDimensions >;

            // getting information about the storage
            typedef typename accessor_t::index_type index_t;

            typedef typename local_domain_t::template get_storage< index_t >::type::value_type storage_t;
            typedef typename get_storage_pointer_accessor< local_domain_t, accessor_t >::type storage_pointer_t;

            // getting information about the metadata
            typedef
                typename boost::mpl::at< metadata_map_t, typename storage_t::storage_info_type >::type metadata_index_t;

            pointer< const typename storage_t::storage_info_type > const metadata_ =
                boost::fusion::at< metadata_index_t >(m_local_domain.m_local_metadata);

            using location_type_t = typename accessor_t::location_type;
            // control your instincts: changing the following
            // int_t to uint_t will prevent GCC from vectorizing (compiler bug)

            const int_t pointer_offset = (m_index[metadata_index_t::value]) +
                                         metadata_->template _index< layout_map_t >(
                                             strides().template get< metadata_index_t::value >(), position_offset);

            return get_raw_value(accessor_t(),
                (data_pointer())[current_storage< (accessor_t::index_type::value == 0),
                    local_domain_t,
                    typename accessor_t::type >::value],
                pointer_offset);
        }
    };

} // namespace gridtools

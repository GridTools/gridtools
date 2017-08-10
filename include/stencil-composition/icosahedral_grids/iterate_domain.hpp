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
#include <type_traits>
#include <boost/type_traits/remove_reference.hpp>
#include "../../common/generic_metafunctions/is_not_same.hpp"
#include "../../common/generic_metafunctions/apply_to_sequence.hpp"
#include "../../common/generic_metafunctions/is_not_same.hpp"
#include "../../common/generic_metafunctions/remove_restrict_reference.hpp"
#include "../../common/generic_metafunctions/variadic_to_vector.hpp"
#include "../../common/generic_metafunctions/variadic_typedef.hpp"
#include "../../common/generic_metafunctions/vector_to_set.hpp"
#include "../../common/array.hpp"
#include "../../common/explode_array.hpp"
#include "../iterate_domain_fwd.hpp"
#include "../location_type.hpp"
#include "../iterate_domain_impl_metafunctions.hpp"
#include "../iterate_domain_aux.hpp"
#include "accessor_metafunctions.hpp"
#include "on_neighbors.hpp"

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
        typedef typename iterate_domain_arguments_t::grid_traits_t grid_traits_t;
        typedef typename iterate_domain_arguments_t::grid_t::grid_topology_t grid_topology_t;
        typedef typename grid_topology_t::default_4d_layout_map_t default_4d_layout_map_t;
        typedef typename iterate_domain_arguments_t::esf_sequence_t esf_sequence_t;

        typedef typename local_domain_t::esf_args esf_args_t;

        typedef backend_traits_from_id< backend_ids_t::s_backend_id > backend_traits_t;
        typedef typename backend_traits_from_id< backend_ids_t::s_backend_id >::template select_iterate_domain_cache<
            iterate_domain_arguments_t >::type iterate_domain_cache_t;

        typedef typename iterate_domain_cache_t::ij_caches_map_t ij_caches_map_t;
        typedef typename iterate_domain_cache_t::all_caches_t all_caches_t;

        GRIDTOOLS_STATIC_ASSERT((is_local_domain< local_domain_t >::value), GT_INTERNAL_ERROR);

        typedef typename local_domain_t::storage_info_ptr_fusion_list storage_info_ptrs_t;
        typedef typename local_domain_t::data_ptr_fusion_map data_ptrs_map_t;

        // the number of different storage metadatas used in the current functor
        static const uint_t N_META_STORAGES = boost::mpl::size< storage_info_ptrs_t >::value;
        // the number of storages  used in the current functor
        static const uint_t N_STORAGES = boost::mpl::size< data_ptrs_map_t >::value;
        // the total number of snapshot (one or several per storage)
        static const uint_t N_DATA_POINTERS =
            total_storages< typename local_domain_t::storage_wrapper_list_t, N_STORAGES >::type::value;

        typedef data_ptr_cached< typename local_domain_t::storage_wrapper_list_t > data_ptr_cached_t;
        typedef strides_cached< N_META_STORAGES - 1, storage_info_ptrs_t > strides_cached_t;

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
                (total_storages< typename LocalD::local_args_type, Accessor::index_t::value >::value);
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
            GRIDTOOLS_STATIC_ASSERT((is_accessor< Arg0 >::value), GT_INTERNAL_ERROR);
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
            typedef typename boost::mpl::and_<
                typename boost::mpl::and_<
                    typename boost::mpl::not_< typename accessor_is_cached< Accessor, all_caches_t >::type >::type,
                    typename boost::mpl::not_< typename accessor_holds_data_field< Accessor >::type >::type >::type,
                typename is_accessor< Accessor >::type >::type type;
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
        data_ptr_cached_t const &RESTRICT data_pointer() const {
            return static_cast< IterateDomainImpl const * >(this)->data_pointer_impl();
        }

        /**
           @brief returns the array of pointers to the raw data
        */
        GT_FUNCTION
        data_ptr_cached_t &RESTRICT data_pointer() {
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
            boost::fusion::for_each(m_local_domain.m_local_data_ptrs,
                assign_storage_ptrs< BackendType,
                                        data_ptr_cached_t,
                                        local_domain_t,
                                        processing_elements_block_size_t,
                                        grid_traits_t >(data_pointer(), m_local_domain.m_local_storage_info_ptrs));
        }

        /**
           @brief recursively assignes all the strides

           copies them from the
           local_domain.m_local_metadata vector, and stores them into an instance of the
           \ref strides_cached class.
         */
        template < typename BackendType, typename Strides >
        GT_FUNCTION void assign_stride_pointers() {
            boost::fusion::for_each(m_local_domain.m_local_storage_info_ptrs,
                assign_strides< BackendType, strides_cached_t, local_domain_t, processing_elements_block_size_t >(
                                        strides()));
        }

        /**@brief method for initializing the index */
        template < ushort_t Coordinate >
        GT_FUNCTION void initialize(uint_t const initial_pos = 0, uint_t const block = 0) {
            boost::fusion::for_each(m_local_domain.m_local_storage_info_ptrs,
                initialize_index_functor< Coordinate,
                                        strides_cached_t,
                                        local_domain_t,
                                        array_index_t,
                                        processing_elements_block_size_t,
                                        grid_traits_t >(strides(), initial_pos, block, m_index));
            static_cast< IterateDomainImpl * >(this)->template initialize_impl< Coordinate >();
            m_grid_position[Coordinate] = initial_pos;
        }

        /**@brief method for incrementing by 1 the index when moving forward along the given direction
           \tparam Coordinate dimension being incremented
           \tparam Execution the policy for the increment (e.g. forward/backward)
         */
        template < ushort_t Coordinate, typename Steps >
        GT_FUNCTION void increment() {
            boost::fusion::for_each(m_local_domain.m_local_storage_info_ptrs,
                increment_index_functor< local_domain_t, Coordinate, strides_cached_t, array_index_t >(
                                        Steps::value, m_index, strides()));
            static_cast< IterateDomainImpl * >(this)->template increment_impl< Coordinate, Steps >();
            m_grid_position[Coordinate] =
                (uint_t)((int_t)m_grid_position[Coordinate] + Steps::value); // suppress warning
        }

        /**@brief method for incrementing the index when moving forward along the given direction

           \param steps_ the increment
           \tparam Coordinate dimension being incremented
         */
        template < ushort_t Coordinate >
        GT_FUNCTION void increment(int_t steps_) {
            boost::fusion::for_each(m_local_domain.m_local_storage_info_ptrs,
                increment_index_functor< local_domain_t, Coordinate, strides_cached_t, array_index_t >(
                                        steps_, m_index, strides()));
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

        template < typename Accessor >
        GT_FUNCTION
            typename boost::disable_if< typename accessor_holds_data_field< Accessor >::type, void * RESTRICT >::type
            get_data_pointer(Accessor const &accessor) const {
            typedef Accessor accessor_t;
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< accessor_t >::value), "Using EVAL is only allowed for an accessor type");
            typedef typename accessor_t::index_t index_t;
            typedef typename local_domain_t::template get_arg< index_t >::type arg_t;
            typedef typename storage_wrapper_elem< arg_t, typename local_domain_t::storage_wrapper_list_t >::type
                storage_wrapper_t;
            typedef typename storage_wrapper_t::storage_t storage_t;
            typedef typename storage_wrapper_t::storage_info_t storage_info_t;
            typedef typename storage_wrapper_t::data_t data_t;

            GRIDTOOLS_STATIC_ASSERT(accessor_t::n_dimensions <= storage_info_t::layout_t::masked_length,
                "Requested accessor index lower than zero. Check that when you define the accessor you specify the "
                "dimenisons which you actually access. e.g. suppose that a storage linked to the accessor ```in``` has "
                "5 dimensions, and thus can be called with in(Dimensions<5>(-1)). Calling in(Dimensions<6>(-1)) brings "
                "you here.");

            return data_pointer().template get< index_t::value >()[0];
        }

        template < typename Accessor >
        GT_FUNCTION
            typename boost::enable_if< typename accessor_holds_data_field< Accessor >::type, void * RESTRICT >::type
            get_data_pointer(Accessor const &accessor) const {

            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");
            typedef typename Accessor::index_t index_t;
            typedef typename local_domain_t::template get_arg< index_t >::type arg_t;

            typedef typename storage_wrapper_elem< arg_t, typename local_domain_t::storage_wrapper_list_t >::type
                storage_wrapper_t;
            typedef typename storage_wrapper_t::storage_t storage_t;
            typedef typename storage_wrapper_t::storage_info_t storage_info_t;
            typedef typename storage_wrapper_t::data_t data_t;

            GRIDTOOLS_STATIC_ASSERT(storage_info_t::layout_t::masked_length + 2 >= Accessor::n_dimensions,
                "the dimension of the accessor exceeds the data field dimension");
            GRIDTOOLS_STATIC_ASSERT(Accessor::n_dimensions != storage_info_t::layout_t::masked_length,
                "The dimension of the data_store_field accessor must be bigger than the storage dimension, you "
                "specified it "
                "equal to the storage dimension");
            GRIDTOOLS_STATIC_ASSERT(Accessor::n_dimensions > storage_info_t::layout_t::masked_length,
                "You specified a too small dimension for the data_store_field");

            const uint_t idx = get_datafield_offset< storage_t >::get(accessor);
            assert(idx < storage_t::num_of_storages && "Out of bounds access when accessing data store field element.");
            return data_pointer().template get< index_t::value >()[idx];
        }

        /** @brief method returning the data pointer of an accessor
            specialization for the accessor placeholders for standard storages

            this method is enabled only if the current placeholder dimension does not exceed the number of space
           dimensions of the storage class.
            I.e., if we are dealing with storages, not with storage lists or data fields (see concepts page for
           definitions)
        */

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
        GT_FUNCTION typename boost::disable_if< typename cache_access_accessor< Accessor >::type,
            typename accessor_return_type< Accessor >::type >::type
        operator()(static_uint< Color >, Accessor const &accessor_) const {
#ifndef NDEBUG
            ASSERT_OR_THROW((check_accessor< grid_traits_t, typename Accessor::extent_t >::apply(accessor_)),
                "Accessor out of bounds.");
#endif
            return get_value(accessor_, get_data_pointer(accessor_));
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
            typedef typename Accessor::index_t index_t;
            typedef typename local_domain_t::template get_arg< index_t >::type arg_t;

            typedef typename storage_wrapper_elem< arg_t, typename local_domain_t::storage_wrapper_list_t >::type
                storage_wrapper_t;
            typedef typename storage_wrapper_t::storage_t storage_t;
            typedef typename storage_wrapper_t::storage_info_t storage_info_t;
            typedef typename storage_wrapper_t::data_t data_t;

            // this index here describes the position of the storage info in the m_index array (can be different to the
            // storage info id)
            typedef typename boost::mpl::find< typename local_domain_t::storage_info_ptr_list,
                const storage_info_t * >::type::pos storage_info_index_t;

            const storage_info_t *storage_info =
                boost::fusion::at< storage_info_index_t >(m_local_domain.m_local_storage_info_ptrs);

            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");

            assert(storage_pointer);
            data_t *RESTRICT real_storage_pointer = static_cast< data_t * >(storage_pointer);
            assert(real_storage_pointer);

            // control your instincts: changing the following
            // int_t to uint_t will prevent GCC from vectorizing (compiler bug)
            const int_t pointer_offset =
                m_index[storage_info_index_t::value] +
                compute_offset< storage_info_t >(strides().template get< storage_info_index_t::value >(), accessor);

#ifndef NDEBUG
            GTASSERT((pointer_oob_check< backend_traits_t,
                processing_elements_block_size_t,
                local_domain_t,
                arg_t,
                grid_traits_t >(storage_info, real_storage_pointer, pointer_offset)));
#endif

            return static_cast< const IterateDomainImpl * >(this)
                ->template get_value_impl<
                    typename iterate_domain< IterateDomainImpl >::template accessor_return_type< Accessor >::type,
                    Accessor,
                    data_t * >(real_storage_pointer, pointer_offset);
        }

        template < typename Accessor, typename StoragePointer >
        GT_FUNCTION typename accessor_return_type< Accessor >::type get_raw_value(
            Accessor const &accessor, StoragePointer &RESTRICT storage_pointer, const uint_t offset) const {
            // getting information about the storage
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");

            typedef typename Accessor::index_t index_t;
            typedef typename local_domain_t::template get_arg< index_t >::type arg_t;

            typedef typename storage_wrapper_elem< arg_t, typename local_domain_t::storage_wrapper_list_t >::type
                storage_wrapper_t;
            typedef typename storage_wrapper_t::storage_t storage_t;
            typedef typename storage_wrapper_t::storage_info_t storage_info_t;
            typedef typename storage_wrapper_t::data_t data_t;

            data_t *RESTRICT real_storage_pointer =
                static_cast< data_t * >(data_pointer().template get< index_t::value >()[0]);

#ifndef NDEBUG
            typedef typename boost::mpl::find< typename local_domain_t::storage_info_ptr_list,
                const storage_info_t * >::type::pos storage_info_index_t;
            const storage_info_t *storage_info =
                boost::fusion::at< storage_info_index_t >(m_local_domain.m_local_storage_info_ptrs);

            GTASSERT((pointer_oob_check< backend_traits_t,
                processing_elements_block_size_t,
                local_domain_t,
                arg_t,
                grid_traits_t >(storage_info, real_storage_pointer, offset)));
#endif
            return static_cast< const IterateDomainImpl * >(this)
                ->template get_value_impl<
                    typename iterate_domain< IterateDomainImpl >::template accessor_return_type< Accessor >::type,
                    Accessor,
                    data_t * >(real_storage_pointer, offset);
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
            using accessor_t = accessor< ID, Intend, LocationType, Extent, FieldDimensions >;
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< accessor_t >::value), "Using EVAL is only allowed for an accessor type");

            // getting information about the storage
            typedef typename accessor_t::index_t index_t;
            typedef typename local_domain_t::template get_arg< index_t >::type arg_t;

            typedef typename storage_wrapper_elem< arg_t, typename local_domain_t::storage_wrapper_list_t >::type
                storage_wrapper_t;
            typedef typename storage_wrapper_t::storage_t storage_t;
            typedef typename storage_wrapper_t::storage_info_t storage_info_t;
            typedef typename storage_wrapper_t::data_t data_t;

            // this index here describes the position of the storage info in the m_index array (can be different to the
            // storage info id)
            typedef typename boost::mpl::find< typename local_domain_t::storage_info_ptr_list,
                const storage_info_t * >::type::pos storage_info_index_t;

            const storage_info_t *storage_info =
                boost::fusion::at< storage_info_index_t >(m_local_domain.m_local_storage_info_ptrs);

            using location_type_t = typename accessor_t::location_type;
            // control your instincts: changing the following
            // int_t to uint_t will prevent GCC from vectorizing (compiler bug)
            const int_t pointer_offset = m_index[storage_info_index_t::value] +
                                         compute_offset< storage_info_t >(
                                             strides().template get< storage_info_index_t::value >(), position_offset);

            return get_raw_value(accessor_t(), data_pointer().template get< index_t::value >()[0], pointer_offset);
        }
    };

} // namespace gridtools

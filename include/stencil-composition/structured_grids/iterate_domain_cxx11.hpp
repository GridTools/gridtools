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

/**@file
   @brief file handling the access to the storage.
   This file implements some of the innermost data access operations of the library and thus it must be highly
   optimized.
   The naming convention used below distinguishes from the following two concepts:

   - a parameter: is a non-space dimension (e.g. time) such that derivatives are taken in the equations along this
   dimension.
   - a dimension: with an abuse of notation will denote any physical scalar field contained in the given storage, e.g. a
   velocity component, the pressure, or the energy. I.e. it is an extra dimension which can appear in the equations only
   derived in space or with respect to the parameter mentioned above.
   - a storage: is an instance of the storage class, and can contain one or more fields and dimensions. Every dimension
   consists of one or several snaphsots of the scalar fields
   (e.g. if the time T is the current dimension, 3 snapshots can be the fields at t, t+1, t+2)
   - a data snapshot: is a pointer to one single snapshot. The snapshots are arranged in the storages on a 1D array,
   regardless of the dimension and snapshot they refer to. The accessor (or offset_tuple) class is
   responsible of computing the correct offests (relative to the given dimension) and address the storages correctly.

   The access to the storage is performed in the following steps:
   - the addresses of the first element of all the data fields in the storages involved in this stencil are saved in an
   array (m_storage_pointers)
   - the index of the storages is saved in another array (m_index)
   - when the functor gets called, the 'offsets' become visible (in the perfect world they could possibly be known at
   compile time). In particular the index is moved to point to the correct address, and the correct data snapshot is
   selected.

   Graphical (ASCII) illustration:

   \verbatim
   ################## Storage #################
   #                ___________\              #
   #                  width    /              #
   #              | |*|*|*|*|*|*|    dim_1    #
   #   dimensions | |*|*|*|          dim_2    #
   #              v |*|*|*|*|*|      dim_3    #
   #                                          #
   #                 ^ ^ ^ ^ ^ ^              #
   #                 | | | | | |              #
   #                 snapshots                #
   #                                          #
   ################## Storage #################
   \endverbatim

*/
#include "../get_data_field_index.hpp"

namespace gridtools {

    /**@brief class managing the memory accesses, indices increment

       This class gets instantiated in the backend-specific code, and has a different implementation for
       each backend (see CRTP pattern). It is instantiated whithin the kernel (e.g. in the device code),
       and drives all the operations which are performed at the innermost level. In particular
       the computation/increment of the useful addresses in memory, given the iteration point,
       the storage placeholders/metadatas and their offsets.
     */
    template < typename IterateDomainImpl >
    struct iterate_domain
        : public iterate_domain_reduction< typename iterate_domain_impl_arguments< IterateDomainImpl >::type > {

        // *************** internal type definitions **************
        typedef typename iterate_domain_impl_arguments< IterateDomainImpl >::type iterate_domain_arguments_t;
        typedef typename iterate_domain_arguments_t::local_domain_t local_domain_t;
        typedef iterate_domain_reduction< iterate_domain_arguments_t > iterate_domain_reduction_t;
        typedef typename iterate_domain_reduction_t::reduction_type_t reduction_type_t;
        typedef typename iterate_domain_arguments_t::processing_elements_block_size_t processing_elements_block_size_t;
        typedef typename iterate_domain_backend_id< IterateDomainImpl >::type backend_id_t;
        typedef typename backend_traits_from_id< backend_id_t::value >::template select_iterate_domain_cache<
            iterate_domain_arguments_t >::type iterate_domain_cache_t;
        typedef typename iterate_domain_cache_t::all_caches_t all_caches_t;
        GRIDTOOLS_STATIC_ASSERT((is_local_domain< local_domain_t >::value), "Internal Error: wrong type");
        // **************** end of internal type definitions
        //***************** types exposed in API
        typedef typename compute_readonly_args_indices< typename iterate_domain_arguments_t::esf_sequence_t >::type
            readonly_args_indices_t;
        typedef typename local_domain_t::esf_args esf_args_t;
        //*****************
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
         * metafunction that computes the return type of all operator() of an accessor.
         *
         * If the temaplate argument is not an accessor ::type is mpl::void_
         *
         */
        template < typename Accessor >
        struct accessor_return_type {
            typedef typename ::gridtools::accessor_return_type_impl< Accessor, iterate_domain_arguments_t >::type type;
        };

        typedef typename local_domain_t::storage_metadata_map metadata_map_t;
        typedef typename local_domain_t::actual_args_type actual_args_type;
        // the number of different storage metadatas used in the current functor
        static const uint_t N_META_STORAGES = boost::mpl::size< metadata_map_t >::value;
        // the number of storages  used in the current functor
        static const uint_t N_STORAGES = boost::mpl::size< actual_args_type >::value;
        // the total number of snapshot (one or several per storage)
        static const uint_t N_DATA_POINTERS = total_storages< actual_args_type,
            boost::mpl::size< typename local_domain_t::mpl_storages >::type::value >::value;

        typedef array< int_t, N_META_STORAGES > array_index_t;

      public:
        typedef array< void * RESTRICT, N_DATA_POINTERS > data_pointer_array_t;
        typedef strides_cached< N_META_STORAGES - 1, typename local_domain_t::storage_metadata_vector_t >
            strides_cached_t;
        // *************** end of type definitions **************

      protected:
        // ******************* members *******************
        local_domain_t const &local_domain;
        array_index_t m_index;
        // ******************* end of members *******************

      public:
        /**
           @brief returns the array of pointers to the raw data as const reference
        */
        GT_FUNCTION
        data_pointer_array_t const &RESTRICT data_pointer() const {
            return static_cast< const IterateDomainImpl * >(this)->data_pointer_impl();
        }

        /**
           @brief returns the array of pointers to the raw data as const reference
        */
        GT_FUNCTION
        array_index_t const &RESTRICT index() const { return m_index; }

      protected:
        /**
           @brief returns the array of pointers to the raw data
        */
        GT_FUNCTION
        data_pointer_array_t &RESTRICT data_pointer() {
            return static_cast< IterateDomainImpl * >(this)->data_pointer_impl();
        }

        /**
           @brief returns the strides
        */
        GT_FUNCTION
        strides_cached_t &RESTRICT strides() { return static_cast< IterateDomainImpl * >(this)->strides_impl(); }

        /**
           @brief returns the strides as const reference
        */
        GT_FUNCTION
        strides_cached_t const &RESTRICT strides() const {
            return static_cast< const IterateDomainImpl * >(this)->strides_impl();
        }

      public:
        /**@brief constructor of the iterate_domain struct

           It assigns the storage pointers to the first elements of
           the data fields (for all the data_fields present in the
           current evaluation), and the indexes to access the data
           fields (one index per storage instance, so that one index
           might be shared among several data fileds)
        */
        GT_FUNCTION
        iterate_domain(local_domain_t const &local_domain_, const reduction_type_t &reduction_initial_value)
            : iterate_domain_reduction_t(reduction_initial_value), local_domain(local_domain_) {}

        /**
           @brief returns a single snapshot in the array of raw data pointers
           \param i index in the array of raw data pointers
        */
        GT_FUNCTION
        const void *data_pointer(ushort_t i) { return (data_pointer())[i]; }

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
                    data_pointer(), local_domain.m_local_args, local_domain.m_local_metadata, EU_id_i, EU_id_j));
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
                processing_elements_block_size_t >(strides(), local_domain.m_local_metadata));
        }

        /**@brief getter for the index array */
        GT_FUNCTION
        void get_index(array< int_t, N_META_STORAGES > &index) const {
            set_index_recur< N_META_STORAGES - 1 >::set(m_index, index);
        }

        /**@brief method for setting the index array
        * This method is responsible of assigning the index for the memory access at
        * the location (i,j,k). Such index is shared among all the fields contained in the
        * same storage class instance, and it is not shared among different storage instances.
        */
        // TODO implement the recursive one, as below, performance is better
        template < typename Value >
        GT_FUNCTION void set_index(array< Value, N_META_STORAGES > const &index) {
            set_index_recur< N_META_STORAGES - 1 >::set(index, m_index);
        }

        GT_FUNCTION
        void set_index(const int index) { set_index_recur< N_META_STORAGES - 1 >::set(index, m_index); }

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
                boost::fusion::as_vector(local_domain.m_local_metadata), Steps::value, m_index, strides()));
            static_cast< IterateDomainImpl * >(this)->template increment_impl< Coordinate, Steps >();
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
                array_index_t >(boost::fusion::as_vector(local_domain.m_local_metadata), steps_, m_index, strides()));
            static_cast< IterateDomainImpl * >(this)->template increment_impl< Coordinate >(steps_);
        }

        /**@brief method for initializing the index */
        template < ushort_t Coordinate >
        GT_FUNCTION void initialize(uint_t const initial_pos = 0, uint_t const block = 0) {
            boost::mpl::for_each< metadata_map_t >(initialize_index_functor< Coordinate,
                strides_cached_t,
                typename boost::fusion::result_of::as_vector< typename local_domain_t::local_metadata_type >::type,
                array_index_t >(
                strides(), boost::fusion::as_vector(local_domain.m_local_metadata), initial_pos, block, m_index));
            static_cast< IterateDomainImpl * >(this)->template initialize_impl< Coordinate >();
        }

        template < typename T >
        GT_FUNCTION void info(T const &x) const {
            local_domain.info(x);
        }

        /**@brief returns the value of the memory at the given address, plus the offset specified by the arg placeholder
           \param arg placeholder containing the storage ID and the offsets
           \param storage_pointer pointer to the first element of the specific data field used
        */
        template < typename Accessor, typename StoragePointer, bool DirectGMemAccess = false >
        GT_FUNCTION typename accessor_return_type< Accessor >::type get_value(
            Accessor const &accessor, StoragePointer const &RESTRICT storage_pointer) const;

        /** specialization for expr_direct_access*/
        template < typename Accessor, typename StoragePointer >
        GT_FUNCTION typename accessor_return_type< Accessor >::type get_value(
            expr_direct_access< Accessor > const &expr, StoragePointer const &RESTRICT storage_pointer) const;

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

            typedef typename Accessor::index_type index_t;
            typedef typename local_domain_t::template get_storage< index_t >::type::value_type storage_t;

            GRIDTOOLS_STATIC_ASSERT(Accessor::n_dim <= storage_t::space_dimensions,
                "requested accessor index lower than zero. Check that when you define the accessor you specify the "
                "dimenisons which you actually access. e.g. suppose that a storage linked to the accessor ```in``` has "
                "5 dimensions, and thus can be called with in(Dimensions<5>(-1)). Calling in(Dimensions<6>(-1)) brings "
                "you here.");

            typedef typename boost::remove_const< typename boost::remove_reference< Accessor >::type >::type acc_t;
            GRIDTOOLS_STATIC_ASSERT((is_accessor< acc_t >::value), "Using EVAL is only allowed for an accessor type");
            return (data_pointer())[current_storage< (acc_t::index_type::value == 0), local_domain_t, acc_t >::value];
        }

        /** @brief method returning the data pointer of an accessor
            specialization for the accessor placeholders for expressions
        */
        template < typename Accessor >
        GT_FUNCTION void *RESTRICT get_data_pointer(expr_direct_access< Accessor > const &accessor) const {

            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");
            return (
                data_pointer())[current_storage< (Accessor::index_type::value == 0), local_domain_t, Accessor >::value];
        }

        /** @brief method returning the data pointer of an accessor
            Specialization for the accessor placeholder for extended storages,
            containg multiple snapshots of data fields with the same dimension and memory layout)

            this method is enabled only if the current placeholder dimension exceeds the number of space dimensions of
           the storage class.
            I.e., if we are dealing with  storage lists or data fields (see concepts page for definitions).
        */
        template < typename Accessor >
        GT_FUNCTION
            typename boost::enable_if< typename accessor_holds_data_field< Accessor >::type, void * RESTRICT >::type
            get_data_pointer(Accessor const &accessor) const {
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");

            typedef typename get_storage_accessor< local_domain_t, Accessor >::type::value_type storage_type;

            // if the following assertion fails you have specified a dimension for the extended storage
            // which does not correspond to the size of the extended placeholder for that storage
            GRIDTOOLS_STATIC_ASSERT(storage_type::space_dimensions + 2 /*max. extra dimensions*/ >= Accessor::n_dim,
                "the dimension of the accessor exceeds the data field dimension");

            GRIDTOOLS_STATIC_ASSERT(Accessor::n_dim != storage_type::space_dimensions,
                "The dimension of the data_field accessor must be bigger than the storage dimension, you specified it "
                "equal to the storage dimension");

            GRIDTOOLS_STATIC_ASSERT(Accessor::n_dim > storage_type::space_dimensions,
                "You specified a too small dimension for the data_field");

            // for the moment the extra dimensionality of the storage is limited to max 2
            //(3 space dim + 2 extra= 5, which gives n_dim==4)
            GRIDTOOLS_STATIC_ASSERT(
                N_DATA_POINTERS > 0, "the total number of snapshots must be larger than 0 in each functor");

            uint_t idx =
                get_data_field_index< storage_type::traits::is_rectangular, Accessor, local_domain_t >::apply(accessor);

            return (data_pointer())[idx];
        }

        /** @brief method called in the Do methods of the functors.

            specialization for the generic accessors placeholders
        */
        template < uint_t I, enumtype::intend Intend >
        GT_FUNCTION typename accessor_return_type< global_accessor< I, Intend > >::type operator()(
            global_accessor< I, Intend > const &accessor) {

            // getting information about the storage
            typedef typename global_accessor< I, Intend >::index_type index_t;

            typedef
                typename get_storage_accessor< local_domain_t, global_accessor< I, Intend > >::type storage_ptr_type;

            storage_ptr_type storage_ = boost::fusion::at< index_t >(local_domain.m_local_args);

            return *storage_;
        }

        /**@brief returns the dimension of the storage corresponding to the given accessor

           Useful to determine the loop bounds, when looping over a dimension from whithin a kernel
         */
        template < ushort_t Coordinate, typename Accessor >
        GT_FUNCTION uint_t get_storage_dim(Accessor) const {

            GRIDTOOLS_STATIC_ASSERT(is_accessor< Accessor >::value, "wrong type");
            typedef typename Accessor::index_type index_t;
            typedef typename local_domain_t::template get_storage< index_t >::type::value_type storage_t;
            // getting information about the metadata
            typedef
                typename boost::mpl::at< metadata_map_t, typename storage_t::storage_info_type >::type metadata_index_t;

            pointer< const typename storage_t::storage_info_type > const metadata_ =
                boost::fusion::at< metadata_index_t >(local_domain.m_local_metadata);

            return metadata_->template dim< Coordinate >();
        }

        /** @brief return a the value in gmem pointed to by an accessor
        */
        template < typename ReturnType, typename StoragePointer >
        GT_FUNCTION ReturnType get_gmem_value(StoragePointer RESTRICT &storage_pointer
            // control your instincts: changing the following
            // int_t to uint_t will prevent GCC from vectorizing (compiler bug)
            ,
            const int_t pointer_offset) const {
            assert(storage_pointer);
            return *(storage_pointer + pointer_offset);
        }

        // some aliases to ease the notation
        template < typename Accessor >
        using cached = typename cache_access_accessor< Accessor >::type;

        /** @brief method called in the Do methods of the functors.

            Specialization for the offset_tuple placeholder (i.e. for extended storages, containg multiple snapshots of
           data fields with the same dimension and memory layout)*/
        template < typename Accessor >
        GT_FUNCTION
            typename boost::enable_if< cached< Accessor >, typename accessor_return_type< Accessor >::type >::type
            operator()(Accessor const &accessor_) {

            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");
            return static_cast< IterateDomainImpl const * >(this)
                ->template get_cache_value_impl< typename accessor_return_type< Accessor >::type >(accessor_);
        }

        template < typename Accessor >
        GT_FUNCTION typename accessor_return_type< Accessor >::type gmem_access(Accessor const &accessor) const {
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");
            GRIDTOOLS_STATIC_ASSERT(
                (Accessor::n_dim > 2), "Accessor with less than 3 dimensions. Did you forget a \"!\"?");

            return get_value< Accessor, void *, true >(accessor, get_data_pointer(accessor));
        }

        template < typename Accessor >
        GT_FUNCTION typename boost::disable_if<
            boost::mpl::or_< cached< Accessor >, boost::mpl::not_< is_accessor< Accessor > > >,
            typename accessor_return_type< Accessor >::type >::type
        operator()(Accessor const &accessor) {
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");
            GRIDTOOLS_STATIC_ASSERT(
                (Accessor::n_dim > 2), "Accessor with less than 3 dimensions. Did you forget a \"!\"?");

            return get_value(accessor, get_data_pointer(accessor));
        }

        template < typename Accessor >
        GT_FUNCTION typename accessor_return_type< Accessor >::type operator()(
            expr_direct_access< Accessor > const &accessor) {
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");
            return get_value(accessor, get_data_pointer(accessor));
        }

        /** @brief method called in the Do methods of the functors

            Overload of the operator() for expressions.
        */
        template < typename... Arguments, template < typename... Args > class Expression >
        GT_FUNCTION auto operator()(Expression< Arguments... > const &arg)
            -> decltype(expressions::evaluation::value(*this, arg)) {

            GRIDTOOLS_STATIC_ASSERT((is_expr< Expression< Arguments... > >::value), "invalid expression");
            return expressions::evaluation::value((*this), arg);
        }

        /** @brief method called in the Do methods of the functors.

            partial specializations for int. Here we do not use the typedef int_t, because otherwise the interface would
           be polluted with casting
            (the user would have to cast all the numbers (-1, 0, 1, 2 .... ) to int_t before using them in the
           expression)*/
        template < typename Argument, template < typename Arg1, int Arg2 > class Expression, int exponent >
        GT_FUNCTION auto operator()(Expression< Argument, exponent > &arg)
            -> decltype(expressions::evaluation::value((*this), arg)) {

            GRIDTOOLS_STATIC_ASSERT((is_expr< Expression< Argument, exponent > >::value), "invalid expression");
            return expressions::evaluation::value((*this), arg);
        }
    };

    //    ################## IMPLEMENTATION ##############################

    /**@brief returns the value of the memory at the given address, plus the offset specified by the arg placeholder
       \param arg placeholder containing the storage ID and the offsets
       \param storage_pointer pointer to the first element of the specific data field used
    */
    template < typename IterateDomainImpl >
    template < typename Accessor, typename StoragePointer, bool DirectGMemAccess >
    GT_FUNCTION typename iterate_domain< IterateDomainImpl >::template accessor_return_type< Accessor >::type
    iterate_domain< IterateDomainImpl >::get_value(
        Accessor const &accessor, StoragePointer const &RESTRICT storage_pointer) const {
        typedef typename iterate_domain< IterateDomainImpl >::template accessor_return_type< Accessor >::type return_t;
        // getting information about the storage
        typedef typename Accessor::index_type index_t;

        typedef typename local_domain_t::template get_storage< index_t >::type::value_type storage_t;
        typedef typename get_storage_pointer_accessor< local_domain_t, Accessor >::type storage_pointer_t;

        GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");

        assert(storage_pointer);
        typename storage_t::value_type *RESTRICT real_storage_pointer =
            static_cast< typename storage_t::value_type * >(storage_pointer);

        assert(real_storage_pointer);
        // getting information about the metadata
        typedef typename boost::mpl::at< metadata_map_t, typename storage_t::storage_info_type >::type metadata_index_t;

        pointer< const typename storage_t::storage_info_type > const metadata_ =
            boost::fusion::at< metadata_index_t >(local_domain.m_local_metadata);
        // getting the value

        // the following assert fails when an out of bound access is observed, i.e. either one of
        // i+offset_i or j+offset_j or k+offset_k is too large.
        // Most probably this is due to you specifying a positive offset which is larger than expected,
        // or maybe you did a mistake when specifying the ranges in the placehoders definition
        GTASSERT(metadata_->size() >
                 m_index[ // Accessor::index_type::value
                     metadata_index_t::value] +
                     metadata_->_index(strides().template get< metadata_index_t::value >(), accessor.offsets()));

        // the following assert fails when an out of bound access is observed,
        // i.e. when some offset is negative and either one of
        // i+offset_i or j+offset_j or k+offset_k is too small.
        // Most probably this is due to you specifying a negative offset which is
        // smaller than expected, or maybe you did a mistake when specifying the ranges
        // in the placehoders definition.
        // If you are running a parallel simulation another common reason for this to happen is
        // the definition of an halo region which is too small in one direction
        // std::cout<<"Storage Index: "<<Accessor::index_type::value<<" + "<<(boost::fusion::at<typename
        // Accessor::index_type>(local_domain.local_args))->_index(arg.template n<Accessor::n_dim>())<<std::endl;
        GTASSERT((int_t)(m_index[metadata_index_t::value]) +
                     metadata_->_index(strides().template get< metadata_index_t::value >(), accessor.offsets()) >=
                 0);

        // control your instincts: changing the following
        // int_t to uint_t will prevent GCC from vectorizing (compiler bug)
        const int_t pointer_offset =
            (m_index[metadata_index_t::value]) +
            metadata_->_index(strides().template get< metadata_index_t::value >(), accessor.offsets());

        if (DirectGMemAccess) {
            return get_gmem_value< return_t >(real_storage_pointer, pointer_offset);
        } else {
            return static_cast< const IterateDomainImpl * >(this)
                ->template get_value_impl<
                    typename iterate_domain< IterateDomainImpl >::template accessor_return_type< Accessor >::type,
                    Accessor,
                    storage_pointer_t >(real_storage_pointer, pointer_offset);
        }
    }

    /** @brief method called in the Do methods of the functors.

        specialization for the expr_direct_access<Accessor> placeholders (high level syntax: '@plch').
        Allows direct access to the storage by only using the offsets
    */
    template < typename IterateDomainImpl >
    template < typename Accessor, typename StoragePointer >
    GT_FUNCTION typename iterate_domain< IterateDomainImpl >::template accessor_return_type< Accessor >::type
    iterate_domain< IterateDomainImpl >::get_value(
        expr_direct_access< Accessor > const &expr, StoragePointer const &RESTRICT storage_pointer) const {
        GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");

        // getting information about the storage
        typedef typename Accessor::index_type index_t;

        typedef typename local_domain_t::template get_storage< index_t >::type::value_type storage_t;

        // getting information about the metadata
        typedef typename boost::mpl::at< metadata_map_t, typename storage_t::storage_info_type >::type metadata_index_t;

        pointer< const typename storage_t::storage_info_type > const metadata_ =
            boost::fusion::at< metadata_index_t >(local_domain.m_local_metadata);

        // error checks
        GTASSERT(metadata_->size() >
                 metadata_->_index(strides().template get< metadata_index_t::value >(), expr.first_operand.offsets()));

        GTASSERT(
            metadata_->_index(strides().template get< metadata_index_t::value >(), expr.first_operand.offsets()) >= 0);

        GRIDTOOLS_STATIC_ASSERT((Accessor::n_dim <= storage_t::storage_info_type::space_dimensions),
            "access out of bound in the storage placeholder (accessor). increase the number of dimensions when "
            "defining the placeholder.");

        // casting the storage pointer from void* to the sotrage value_type
        typename storage_t::value_type *RESTRICT real_storage_pointer =
            static_cast< typename storage_t::value_type * >(storage_pointer);

        // returning the value without adding the m_index
        return *(real_storage_pointer +
                 metadata_->_index(strides().template get< metadata_index_t::value >(), expr.first_operand.offsets()));
    }

} // namespace gridtools

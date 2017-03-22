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
   - when the functor gets called, the 'offsets' become visible (in the perfect worls they could possibly be known at
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

        // *************** type definitions **************

        typedef typename iterate_domain_impl_arguments< IterateDomainImpl >::type iterate_domain_arguments_t;
        typedef typename iterate_domain_arguments_t::local_domain_t local_domain_t;

        typedef iterate_domain_reduction< iterate_domain_arguments_t > iterate_domain_reduction_t;
        typedef typename iterate_domain_reduction_t::reduction_type_t reduction_type_t;

        typedef typename iterate_domain_arguments_t::processing_elements_block_size_t processing_elements_block_size_t;
        // sequence of args types which are readonly through all ESFs/MSSs
        typedef typename compute_readonly_args_indices< typename iterate_domain_arguments_t::esf_sequence_t >::type
            readonly_args_indices_t;

        typedef typename local_domain_t::esf_args esf_args_t;
        typedef typename iterate_domain_backend_id< IterateDomainImpl >::type backend_id_t;
        typedef typename backend_traits_from_id< backend_id_t::value >::template select_iterate_domain_cache<
            iterate_domain_arguments_t >::type iterate_domain_cache_t;
        typedef typename iterate_domain_cache_t::all_caches_t all_caches_t;

        GRIDTOOLS_STATIC_ASSERT((is_local_domain< local_domain_t >::value), GT_INTERNAL_ERROR);

        /**
         * metafunction that determines if a given accessor is associated with an placeholder holding a data field
         */
        template < typename Accessor >
        struct accessor_holds_data_field {
            typedef typename boost::mpl::eval_if< is_accessor< Accessor >,
                arg_holds_data_field_h< get_arg_from_accessor< Accessor, iterate_domain_arguments_t > >,
                boost::mpl::identity< boost::mpl::false_ > >::type type;
        };

        /**
         * metafunction that determines if a given accessor is associated with an arg holding a data field
         * and the parameter refers to a storage in main memory (i.e. is not cached)
         */
        template < typename Accessor, typename CachesMap >
        struct mem_access_with_data_field_accessor {
            typedef typename boost::mpl::and_<
                typename boost::mpl::not_< typename accessor_is_cached< Accessor, CachesMap >::type >::type,
                typename accessor_holds_data_field< Accessor >::type >::type type;
        };

        /**
         * metafunction that determines if a given accessor is associated with an arg holding a
         * standard field (i.e. not a data field)
         * and the parameter refers to a storage in main memory (i.e. is not cached)
         */
        template < typename Accessor, typename CachesMap >
        struct mem_access_with_standard_accessor {
            typedef typename boost::mpl::and_<
                typename boost::mpl::and_<
                    typename boost::mpl::not_< typename accessor_is_cached< Accessor, CachesMap >::type >::type,
                    typename boost::mpl::not_< typename accessor_holds_data_field< Accessor >::type >::type >::type,
                typename is_accessor< Accessor >::type > type;
        };

        /**
         * metafunction that determines if a given accessor is associated with an arg that is cached
         */
        template < typename Accessor, typename CachesMap >
        struct cache_access_accessor {
            typedef typename accessor_is_cached< Accessor, CachesMap >::type type;
        };

        /**
         * metafunction that computes the return type of all operator() of an accessor.
         *
         * If the temaplate argument is not an accessor ::type is mpl::void_
         *
         */
        template < typename Accessor >
        struct accessor_return_type {
            typedef typename accessor_return_type_impl< Accessor, iterate_domain_arguments_t >::type type;
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
            GRIDTOOLS_STATIC_ASSERT((is_strides_cached< Strides >::value), GT_INTERNAL_ERROR);
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
        template < typename Accessor, typename StoragePointer >
        GT_FUNCTION typename accessor_return_type< Accessor >::type get_value(
            Accessor const &accessor, StoragePointer const &RESTRICT storage_pointer) const;

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
            typedef typename boost::remove_const< typename boost::remove_reference< Accessor >::type >::type acc_t;
            GRIDTOOLS_STATIC_ASSERT((is_accessor< acc_t >::value), "Using EVAL is only allowed for an accessor type");
            return (data_pointer())
                [current_storage< (acc_t::index_t::value == 0), local_domain_t, typename acc_t::type >::value];
        }

#ifdef CXX11_ENABLED
        /** @brief method returning the data pointer of an accessor
            specialization for the accessor placeholders for expressions
        */
        template < typename Accessor >
        GT_FUNCTION void *get_data_pointer(expr_direct_access< Accessor > const &accessor) const {
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");
            return (
                data_pointer())[current_storage< (Accessor::index_t::value == 0), local_domain_t, Accessor >::value];
        }
#endif

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

            typedef typename get_storage_accessor< local_domain_t, Accessor >::type storage_type;

            // if the following assertion fails you have specified a dimension for the extended storage
            // which does not correspond to the size of the extended placeholder for that storage
            GRIDTOOLS_STATIC_ASSERT(
                storage_type::space_dimensions + 2 /*max. extra dimensions*/ >= Accessor::n_dimensions,
                "the dimension of the accessor exceeds the data field dimension");

            // for the moment the extra dimensionality of the storage is limited to max 2
            //(3 space dim + 2 extra= 5, which gives n_dimensions==4)
            GRIDTOOLS_STATIC_ASSERT(
                N_DATA_POINTERS > 0, "the total number of snapshots must be larger than 0 in each functor");
            GRIDTOOLS_STATIC_ASSERT(Accessor::n_dimensions <= storage_type::storage_info_type::space_dimensions,
                "access out of bound in the storage placeholder (accessor). increase the number of dimensions when "
                "defining the placeholder.");

            GRIDTOOLS_STATIC_ASSERT((storage_type::traits::n_fields % storage_type::traits::n_width == 0),
                "You specified a non-rectangular field: if you need to use a non-rectangular field the constexpr "
                "version of the accessors have to be used (so that the current position in the field is computed at "
                "compile time). This is achieved by using, e.g., instead of \n\n eval(field(dimension<5>(2))); \n\n "
                "the following expression: \n\n typedef alias<field, dimension<5> >::set<2> z_field; \n "
                "eval(z_field()); \n");

            // dimension/snapshot offsets must be non negative
            GTASSERT(accessor.template get< 0 >() >= 0);
            GTASSERT(
                (Accessor::n_dimensions <= storage_type::space_dimensions + 1) || (accessor.template get< 1 >() >= 0));
            // std::cout<<" offsets: "<<arg.template get<0>()<<" , "<<arg.template get<1>()<<" , "<<arg.template
            // get<2>()<<" , "<<std::endl;

            return (data_pointer())
                [(Accessor::n_dimensions <= storage_type::space_dimensions + 1 ? // static if
                         accessor.template get< 0 >()
                                                                               : // offset for the current dimension
                         accessor.template get< 1 >()                            // offset for the current snapshot
                             // limitation to "rectangular" vector fields for non-static fields dimensions
                             +
                             accessor.template get< 0 >() // select the dimension
                                 *
                                 storage_type::traits::n_width // stride of the current dimension inside the vector of
                                                               // storages
                     )
                    //+ the offset of the other extra dimension
                    +
                    current_storage< (Accessor::index_t::value == 0), local_domain_t, Accessor >::value];
        }

        /** @brief method called in the Do methods of the functors.

            specialization for the generic accessors placeholders
        */
        template < uint_t I, enumtype::intend Intend >
        GT_FUNCTION typename accessor_return_type< global_accessor< I, Intend > >::type operator()(
            global_accessor< I, Intend > const &accessor) const {

            // getting information about the storage
            typedef typename global_accessor< I, Intend >::index_t index_t;

            typedef
                typename get_storage_accessor< local_domain_t, global_accessor< I, Intend > >::type storage_ptr_type;

            storage_ptr_type storage_ = boost::fusion::at< index_t >(local_domain.m_local_args);

            return *storage_;
        }
#ifdef CXX11_ENABLED
        /** @brief method called in the Do methods of the functors.
            specialization for the expr_direct_access<accessor> placeholders
        */
        template < typename Accessor >
        GT_FUNCTION typename accessor_return_type< Accessor >::type operator()(
            expr_direct_access< Accessor > const &accessor) const {
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");

            return get_value(accessor, get_data_pointer(accessor));
        }

        /**@brief returns the dimension of the storage corresponding to the given accessor

           Useful to determine the loop bounds, when looping over a dimension from whithin a kernel
         */
        template < ushort_t Coordinate, typename Accessor >
        GT_FUNCTION uint_t get_storage_dim(Accessor) const {

            GRIDTOOLS_STATIC_ASSERT(is_accessor< Accessor >::value, GT_INTERNAL_ERROR);
            typedef typename Accessor::index_t index_t;
            typedef typename local_domain_t::template get_storage< index_t >::type::value_type storage_t;
            // getting information about the metadata
            typedef
                typename boost::mpl::at< metadata_map_t, typename storage_t::storage_info_type >::type metadata_index_t;

            pointer< const typename storage_t::storage_info_type > const metadata_ =
                boost::fusion::at< metadata_index_t >(local_domain.m_local_metadata);

            return metadata_->template dim< Coordinate >();
        }
#endif // CXX11_ENABLED

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

        /** @brief method called in the Do methods of the functors.
            specialization for the accessor placeholders

            this method is enabled only if the current placeholder dimension does not exceed the number of space
           dimensions of the storage class.
            I.e., if we are dealing with storages, not with storage lists or data fields (see concepts page for
           definitions)
        */
        template < typename Accessor >
        GT_FUNCTION
            typename boost::enable_if< typename mem_access_with_standard_accessor< Accessor, all_caches_t >::type,
                typename accessor_return_type< Accessor >::type >::type
            operator()(Accessor const &accessor) const {

            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");
            GRIDTOOLS_STATIC_ASSERT(
                (Accessor::n_dimensions > 2), "Accessor with less than 3 dimensions. Did you forget a \"!\"?");
            return get_value(accessor, get_data_pointer(accessor));
        }

        template < typename Accessor >
        GT_FUNCTION typename boost::enable_if< typename cache_access_accessor< Accessor, all_caches_t >::type,
            typename accessor_return_type< Accessor >::type >::type
        operator()(Accessor const &accessor_) const {
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");
            GRIDTOOLS_STATIC_ASSERT(
                (Accessor::n_dimensions > 2), "Accessor with less than 3 dimensions. Did you forget a \"!\"?");
            return static_cast< IterateDomainImpl const * >(this)
                ->template get_cache_value_impl< typename accessor_return_type< Accessor >::type >(accessor_);
        }

        /** @brief method called in the Do methods of the functors.
            Specialization for the accessor placeholder (i.e. for extended storages, containg multiple snapshots of data
           fields with the same dimension and memory layout)

            this method is enabled only if the current placeholder dimension exceeds the number of space dimensions of
           the storage class.
            I.e., if we are dealing with  storage lists or data fields (see concepts page for definitions).
            TODO: This and the above version will be eventually merged.
        */
        template < typename Accessor >
        GT_FUNCTION
            typename boost::enable_if< typename mem_access_with_data_field_accessor< Accessor, all_caches_t >::type,
                typename accessor_return_type< Accessor >::type >::type
            operator()(Accessor const &accessor) const;

#ifdef CXX11_ENABLED

        /** @brief method called in the Do methods of the functors.

            specialization for the expr_direct_access<Accessor> placeholders (high level syntax: '@plch').
            Allows direct access to the storage by only using the offsets
        */
        template < typename Accessor, typename StoragePointer >
        GT_FUNCTION typename accessor_return_type< Accessor >::type get_value(
            expr_direct_access< Accessor > const &accessor, StoragePointer const &RESTRICT storage_pointer) const;

        /** @brief method called in the Do methods of the functors. */
        template < typename... Arguments,
            template < typename... Args > class Expression,
            typename boost::enable_if< is_expr< Expression< Arguments... > >, int >::type = 0 >
        GT_FUNCTION auto operator()(Expression< Arguments... > const &arg) const
            -> decltype(expressions::evaluation::value(*this, arg)) {
            // arg.to_string();
            return expressions::evaluation::value((*this), arg);
        }

        template < typename Argument, template < typename Arg1, int Arg2 > class Expression, int int_argument >
        GT_FUNCTION auto operator()(Expression< Argument, int_argument > const &arg) const
            -> decltype(expressions::evaluation::value((*this), arg)) {

            GRIDTOOLS_STATIC_ASSERT((is_expr< Expression< Argument, int_argument > >::value), "invalid expression");
            return expressions::evaluation::value((*this), arg);
        }

#endif // CXX11_ENABLED
    };

    //    ################## IMPLEMENTATION ##############################

    /**@brief returns the value of the memory at the given address, plus the offset specified by the arg placeholder
       \param arg placeholder containing the storage ID and the offsets
       \param storage_pointer pointer to the first element of the specific data field used
    */
    template < typename IterateDomainImpl >
    template < typename Accessor, typename StoragePointer >
    GT_FUNCTION typename iterate_domain< IterateDomainImpl >::template accessor_return_type< Accessor >::type
    iterate_domain< IterateDomainImpl >::get_value(
        Accessor const &accessor, StoragePointer const &RESTRICT storage_pointer) const {

        // getting information about the storage
        typedef typename Accessor::index_t index_t;

        typedef typename local_domain_t::template get_storage< index_t >::type::value_type storage_t;
        typedef typename get_storage_pointer_accessor< local_domain_t, Accessor >::type storage_pointer_t;

        GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");

        typename storage_t::value_type *RESTRICT real_storage_pointer =
            static_cast< typename storage_t::value_type * >(storage_pointer);

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
                 m_index[ // Accessor::index_t::value
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
        // std::cout<<"Storage Index: "<<Accessor::index_t::value<<" + "<<(boost::fusion::at<typename
        // Accessor::index_t>(local_domain.local_args))->_index(arg.template n<Accessor::n_dimensions>())<<std::endl;
        GTASSERT((int_t)(m_index[metadata_index_t::value]) +
                     metadata_->_index(strides().template get< metadata_index_t::value >(), accessor.offsets()) >=
                 0);

        // control your instincts: changing the following
        // int_t to uint_t will prevent GCC from vectorizing (compiler bug)
        const int_t pointer_offset =
            (m_index[metadata_index_t::value]) +
            metadata_->_index(strides().template get< metadata_index_t::value >(), accessor.offsets());

        return static_cast< const IterateDomainImpl * >(this)
            ->template get_value_impl<
                typename iterate_domain< IterateDomainImpl >::template accessor_return_type< Accessor >::type,
                Accessor,
                storage_pointer_t >(real_storage_pointer, pointer_offset);
    }

#if defined(CXX11_ENABLED)

    /** @brief method called in the Do methods of the functors.
        Specialization for the offset_tuple placeholder (i.e. for extended storages, containg multiple snapshots of data
       fields with the same dimension and memory layout)
       NOTE: this part will be removed when C++03 versions will be no longer supported
    */
    template < typename IterateDomainImpl >
    template < typename Accessor >
    GT_FUNCTION typename boost::enable_if<
        typename iterate_domain< IterateDomainImpl >::template mem_access_with_data_field_accessor< Accessor,
            typename iterate_domain< IterateDomainImpl >::all_caches_t >::type,
        typename iterate_domain< IterateDomainImpl >::template accessor_return_type< Accessor >::type >::type
        iterate_domain< IterateDomainImpl >::
        operator()(Accessor const &accessor) const {

        GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");

#ifdef PEDANTIC // storages may have less than 3 dimensions in non pedantic mode
        GRIDTOOLS_STATIC_ASSERT(
            (Accessor::n_dimensions >= 4), "Accessor with less than 3 dimensions. Did you forget a \"!\"?");
#endif

        // getting information about the storage
        typedef typename Accessor::index_t index_t;

        typedef typename local_domain_t::template get_storage< index_t >::type::value_type storage_t;

        typedef typename storage_t::storage_info_type metadata_t;
        // if the following assertion fails you have specified a dimension for the extended storage
        // which does not correspond to the size of the extended placeholder for that storage
        GRIDTOOLS_STATIC_ASSERT(metadata_t::space_dimensions + 2 /*max. extra dimensions*/ >= Accessor::n_dimensions,
            "the dimension of the accessor exceeds the data field dimension");

        // for the moment the extra dimensionality of the storage is limited to max 2
        //(3 space dim + 2 extra= 5, which gives n_dimensions==4)
        GRIDTOOLS_STATIC_ASSERT(
            N_DATA_POINTERS > 0, "the total number of snapshots must be larger than 0 in each functor");

        GRIDTOOLS_STATIC_ASSERT((storage_t::traits::n_fields % storage_t::traits::n_width == 0),
            "You specified a non-rectangular field: if you need to use a non-rectangular field the constexpr version "
            "of the accessors have to be used (so that the current position in the field is computed at compile time). "
            "This is achieved by using, e.g., instead of \n\n eval(field(dimension<5>(2))); \n\n the following "
            "expression: \n\n typedef alias<field, dimension<5> >::set<2> z_field; \n eval(z_field()); \n");
        GRIDTOOLS_STATIC_ASSERT(
            (storage_t::traits::n_width > 0), "did you define a field dimension with 0 snapshots??");

        // std::cout<<" offsets: "<<accessor.template get<0>()<<" , "<<accessor.template get<1>()<<" ,
        // "<<accessor.template get<2>()<<" , "<<std::endl;

        // dimension/snapshot offsets must be non negative
        GTASSERT(accessor.template get< 0 >() >= 0);
        GTASSERT((Accessor::n_dimensions <= metadata_t::space_dimensions + 1) || (accessor.template get< 1 >() >= 0));

        // snapshot access out of bounds
        GTASSERT((Accessor::n_dimensions > metadata_t::space_dimensions + 1) ||
                 accessor.template get< 0 >() < storage_t::traits::n_width);
        // snapshot access out of bounds
        GTASSERT((Accessor::n_dimensions <= metadata_t::space_dimensions + 1) ||
                 accessor.template get< 1 >() < storage_t::traits::n_width);
        // dimension access out of bounds
        GTASSERT((Accessor::n_dimensions <= metadata_t::space_dimensions + 1) ||
                 accessor.template get< 0 >() < storage_t::traits::n_dimensions);

        GRIDTOOLS_STATIC_ASSERT(Accessor::n_dimensions <= storage_t::space_dimensions + 2,
            "requested accessor index lower than zero. Check that when you define the accessor you specify the "
            "dimenisons which you actually access. e.g. suppose that a storage linked to the accessor ```in``` has 5 "
            "dimensions, and thus can be called with in(Dimensions<5>(-1)). Calling in(Dimensions<6>(-1)) brings you "
            "here.");

        return get_value(accessor,
            (data_pointer())[(Accessor::n_dimensions <= metadata_t::space_dimensions + 1
                                     ?                              // static if
                                     accessor.template get< 0 >()   // offset for the current dimension
                                     : accessor.template get< 1 >() // offset for the current snapshot
                                           // limitation to "rectangular" vector fields for non-static fields dimensions
                                           +
                                           accessor.template get< 0 >() // select the dimension
                                               *
                                               storage_t::traits::n_width // stride of the current dimension inside the
                                                                          // vector of storages
                                 )
                             //+ the offset of the other extra dimension
                             +
                             current_storage< (Accessor::index_t::value == 0), local_domain_t, Accessor >::value]);
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
        typedef typename Accessor::index_t index_t;

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
#endif // defined (CXX11_ENABLED)

} // namespace gridtools

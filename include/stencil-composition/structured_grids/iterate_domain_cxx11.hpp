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

#include "../../common/gt_assert.hpp"

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

namespace gridtools {

    template < typename >
    struct iterate_domain;

    namespace advanced {
        template < typename IDomain >
        inline typename iterate_domain< IDomain >::data_ptr_cached_t &RESTRICT get_iterate_domain_data_pointer(
            iterate_domain< IDomain > &id) {
            return id.data_pointer();
        }

    } // namespace advanced

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
        typedef typename iterate_domain_arguments_t::grid_traits_t grid_traits_t;
        typedef typename iterate_domain_arguments_t::processing_elements_block_size_t processing_elements_block_size_t;
        typedef typename iterate_domain_backend_id< IterateDomainImpl >::type backend_id_t;
        typedef backend_traits_from_id< backend_id_t::value > backend_traits_t;
        typedef typename backend_traits_from_id< backend_id_t::value >::template select_iterate_domain_cache<
            iterate_domain_arguments_t >::type iterate_domain_cache_t;
        typedef typename iterate_domain_cache_t::all_caches_t all_caches_t;
        GRIDTOOLS_STATIC_ASSERT((is_local_domain< local_domain_t >::value), GT_INTERNAL_ERROR);

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

        typedef typename local_domain_t::storage_info_ptr_fusion_list storage_info_ptrs_t;
        typedef typename local_domain_t::data_ptr_fusion_map data_ptrs_map_t;
        // the number of different storage metadatas used in the current functor
        static const uint_t N_META_STORAGES = boost::mpl::size< storage_info_ptrs_t >::value;
        // the number of storages  used in the current functor
        static const uint_t N_STORAGES = boost::mpl::size< data_ptrs_map_t >::value;
        // the total number of snapshot (one or several per storage)
        static const uint_t N_DATA_POINTERS =
            total_storages< typename local_domain_t::storage_wrapper_list_t, N_STORAGES >::type::value;

      public:
        typedef data_ptr_cached< typename local_domain_t::storage_wrapper_list_t > data_ptr_cached_t;
        typedef strides_cached< N_META_STORAGES - 1, storage_info_ptrs_t > strides_cached_t;
        typedef array< int_t, N_META_STORAGES > array_index_t;
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
        data_ptr_cached_t const &RESTRICT data_pointer() const {
            return static_cast< const IterateDomainImpl * >(this)->data_pointer_impl();
        }

      protected:
        /**
           @brief returns the strides as const reference
        */
        GT_FUNCTION
        strides_cached_t const &RESTRICT strides() const {
            return static_cast< const IterateDomainImpl * >(this)->strides_impl();
        }

        /**
           @brief returns the strides
        */
        GT_FUNCTION
        strides_cached_t &RESTRICT strides() { return static_cast< IterateDomainImpl * >(this)->strides_impl(); }

        /**
           @brief returns the array of pointers to the raw data
        */
        GT_FUNCTION
        data_ptr_cached_t &RESTRICT data_pointer() {
            return static_cast< IterateDomainImpl * >(this)->data_pointer_impl();
        }

        friend data_ptr_cached_t &RESTRICT advanced::get_iterate_domain_data_pointer< IterateDomainImpl >(
            iterate_domain &);

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
            : iterate_domain_reduction_t(reduction_initial_value), local_domain(local_domain_), m_index{
                                                                                                    0,
                                                                                                } {}

        /** This functon set the addresses of the data values  before the computation
            begins.

            The EU stands for ExecutionUnit (thich may be a thread or a group of
            threasd. There are potentially two ids, one over i and one over j, since
            our execution model is parallel on (i,j). Defaulted to 1.
        */
        template < typename BackendType >
        GT_FUNCTION void assign_storage_pointers() {
            boost::fusion::for_each(local_domain.m_local_data_ptrs,
                assign_storage_ptrs< BackendType,
                                        data_ptr_cached_t,
                                        local_domain_t,
                                        processing_elements_block_size_t,
                                        grid_traits_t >(data_pointer(), local_domain.m_local_storage_info_ptrs));
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
            boost::fusion::for_each(local_domain.m_local_storage_info_ptrs,
                assign_strides< BackendType, strides_cached_t, local_domain_t, processing_elements_block_size_t >(
                                        strides()));
        }

        GT_FUNCTION array_index_t const &index() const { return m_index; }

        /**@brief method for setting the index array
        * This method is responsible of assigning the index for the memory access at
        * the location (i,j,k). Such index is shared among all the fields contained in the
        * same storage class instance, and it is not shared among different storage instances.
        */
        GT_FUNCTION void set_index(array_index_t const &index) { m_index = index; }

        GT_FUNCTION void reset_index() { m_index = array_index_t{}; }

        /**@brief method for incrementing by 1 the index when moving forward along the given direction
           \tparam Coordinate dimension being incremented
           \tparam Execution the policy for the increment (e.g. forward/backward)
         */
        template < ushort_t Coordinate, typename Steps >
        GT_FUNCTION void increment() {
            boost::fusion::for_each(local_domain.m_local_storage_info_ptrs,
                increment_index_functor< local_domain_t, Coordinate, strides_cached_t, array_index_t >(
                                        Steps::value, m_index, strides()));
            static_cast< IterateDomainImpl * >(this)->template increment_impl< Coordinate, Steps >();
        }

        /**@brief method for incrementing the index when moving forward along the given direction

           \param steps_ the increment
           \tparam Coordinate dimension being incremented
         */
        template < ushort_t Coordinate >
        GT_FUNCTION void increment(int_t steps_) {
            boost::fusion::for_each(local_domain.m_local_storage_info_ptrs,
                increment_index_functor< local_domain_t, Coordinate, strides_cached_t, array_index_t >(
                                        steps_, m_index, strides()));
            static_cast< IterateDomainImpl * >(this)->template increment_impl< Coordinate >(steps_);
        }

        /**@brief method for initializing the index */
        template < ushort_t Coordinate >
        GT_FUNCTION void initialize(uint_t const initial_pos = 0, uint_t const block = 0) {
            boost::fusion::for_each(local_domain.m_local_storage_info_ptrs,
                initialize_index_functor< Coordinate,
                                        strides_cached_t,
                                        local_domain_t,
                                        array_index_t,
                                        processing_elements_block_size_t,
                                        grid_traits_t >(strides(), initial_pos, block, m_index));
            static_cast< IterateDomainImpl * >(this)->template initialize_impl< Coordinate >();
        }

        template < typename T >
        GT_FUNCTION void info(T const &x) const {
            local_domain.info(x);
        }

        /**@brief returns the value of the memory at the given address, plus the offset specified by the arg placeholder
           \param accessor Accessor pass to the evaluator
           \param storage_pointer pointer to the first element of the specific data field used
           \tparam DirectGMemAccess selects a direct access to main memory for the given accessor, ignoring if the
           parameter is being cached using software managed cache syntax
        */
        template < typename Accessor, typename StoragePointer, bool DirectGMemAccess = false >
        GT_FUNCTION typename accessor_return_type< Accessor >::type get_value(
            Accessor const &accessor, StoragePointer const &RESTRICT storage_pointer) const;

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
            typedef typename Accessor::index_t index_t;
            typedef typename local_domain_t::template get_data_store< index_t >::type::storage_info_t storage_info_t;

            GRIDTOOLS_STATIC_ASSERT(Accessor::n_dimensions <= storage_info_t::layout_t::masked_length,
                "requested accessor index lower than zero. Check that when you define the accessor you specify the "
                "dimenisons which you actually access. e.g. suppose that a storage linked to the accessor ```in``` has "
                "5 dimensions, and thus can be called with in(Dimensions<5>(-1)). Calling in(Dimensions<6>(-1)) brings "
                "you here.");

            typedef typename boost::remove_const< typename boost::remove_reference< Accessor >::type >::type acc_t;
            GRIDTOOLS_STATIC_ASSERT((is_accessor< acc_t >::value), "Using EVAL is only allowed for an accessor type");
            return data_pointer().template get< index_t::value >()[0];
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
            typedef typename Accessor::index_t index_t;
            typedef typename local_domain_t::template get_arg< index_t >::type arg_t;

            typedef typename storage_wrapper_elem< arg_t, typename local_domain_t::storage_wrapper_list_t >::type
                storage_wrapper_t;
            typedef typename storage_wrapper_t::data_store_t data_store_t;
            typedef typename storage_wrapper_t::storage_info_t storage_info_t;
            typedef typename storage_wrapper_t::data_t data_t;

            GRIDTOOLS_STATIC_ASSERT(Accessor::n_dimensions == storage_info_t::layout_t::masked_length + 2,
                "The dimension of the data_store_field accessor must be equals to storage dimension + 2 (component and "
                "snapshot)");

            const int_t idx = get_datafield_offset< data_store_t >::get(accessor);
#ifdef CUDA8
            assert(
                idx < data_store_t::num_of_storages && "Out of bounds access when accessing data store field element.");
#endif
            return data_pointer().template get< index_t::value >()[idx];
        }

        /**@brief helper function that given an input in_ and a tuple t_ calls in_.operator() with the elements of the
           tuple as arguments.

           For example, if the tuple is an accessor containing the offsets 1,2,3, and the
           input is a storage st_, this function returns st_(1,2,3).

           \param container the input class
           \param tuple the tuple
         */
        template < typename Container, typename Tuple, uint_t... Ids >
        GT_FUNCTION auto static tuple_to_container(
            Container &&container_, Tuple const &tuple_, gt_integer_sequence< uint_t, Ids... >)
            -> decltype(container_(boost::fusion::at_c< Ids >(tuple_)...)) {
            return container_(boost::fusion::at_c< Ids >(tuple_)...);
        }

        template < typename Acc, typename... Args >
        using ret_t = typename boost::remove_reference< decltype(tuple_to_container(
            std::declval< typename get_storage_accessor< local_domain_t, Acc >::type::storage_t::data_t >(),
            std::declval< global_accessor_with_arguments< Acc, Args... > >().get_arguments(),
            make_gt_integer_sequence< uint_t, sizeof...(Args) >())) >::type;

        /** @brief method called in the Do methods of the functors.

            specialization for the generic accessors placeholders with arguments
        */
        template < typename Acc, typename... Args >
        GT_FUNCTION auto operator()(global_accessor_with_arguments< Acc, Args... > const &accessor) const
            -> ret_t< Acc, Args... >

        {

            typedef typename Acc::index_t index_t;
            auto storage_ = boost::fusion::at< index_t >(local_domain.m_local_data_ptrs).second;

            return tuple_to_container(
                **storage_.data(), accessor.get_arguments(), make_gt_integer_sequence< uint_t, sizeof...(Args) >());
        }

        /**@brief returns the dimension of the storage corresponding to the given accessor

           Useful to determine the loop bounds, when looping over a dimension from whithin a kernel
         */
        template < ushort_t Coordinate, typename Accessor >
        GT_FUNCTION uint_t get_storage_dim(Accessor) const {

            GRIDTOOLS_STATIC_ASSERT(is_accessor< Accessor >::value, GT_INTERNAL_ERROR);
            typedef typename Accessor::index_type index_t;
            typedef typename local_domain_t::template get_data_store< index_t >::type::storage_info_t storage_info_t;
            typedef typename boost::mpl::find< typename local_domain_t::storage_info_ptr_list,
                const storage_info_t * >::type::pos storage_info_index_t;
            return boost::fusion::at< storage_info_index_t >(local_domain.m_local_storage_info_ptrs)
                ->template dim< Coordinate >();
        }

        /** @brief return a the value in gmem pointed to by a base storage pointer and an offset
         * \param storage_pointer base address to gmem
         * \param offset to compose the address being access
        */
        template < typename ReturnType,
            typename StoragePointer,
            typename T = typename boost::enable_if_c< boost::is_pointer< StoragePointer >::type::value >::type >
        GT_FUNCTION ReturnType get_gmem_value(StoragePointer RESTRICT &storage_pointer
            // control your instincts: changing the following
            // int_t to uint_t will prevent GCC from vectorizing (compiler bug)
            ,
            const int_t pointer_offset) const {
#ifdef CUDA8
            assert(storage_pointer);
#endif
            return *(storage_pointer + pointer_offset);
        }

        // some aliases to ease the notation
        template < typename Accessor >
        using cached = typename cache_access_accessor< Accessor >::type;

        /** @brief method called in the Do methods of the functors.

            specialization for the generic accessors placeholders
        */
        template < uint_t I, enumtype::intend Intend >
        GT_FUNCTION typename accessor_return_type< global_accessor< I, Intend > >::type operator()(
            global_accessor< I, Intend > const &accessor) {
            typedef typename accessor_return_type< global_accessor< I, Intend > >::type return_t;
            typedef typename global_accessor< I, Intend >::index_t index_t;
            return *static_cast< return_t * >(data_pointer().template get< index_t::value >()[0]);
        }

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

        /**
         * @brief direct access for an accessor to main memory. No dispatch to a corresponding scratch-pad is performed
         */
        template < typename Accessor,
            typename T = typename boost::enable_if_c<
                is_accessor< typename boost::remove_const< Accessor >::type >::type::value >::type >
        GT_FUNCTION typename accessor_return_type< Accessor >::type get_gmem_value(Accessor const &accessor) const {
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");
            GRIDTOOLS_STATIC_ASSERT(
                (Accessor::n_dimensions > 2), "Accessor with less than 3 dimensions. Did you forget a \"!\"?");

            return get_value< Accessor, void *, true >(accessor, get_data_pointer(accessor));
        }

        /**
         * @brief returns the value pointed by an accessor in case the value is a normal accessor (not global accessor
         * nor expression)
         * and is not cached (i.e. is accessing main memory)
         */
        template < typename Accessor >
        GT_FUNCTION typename boost::disable_if< boost::mpl::or_< cached< Accessor >,
                                                    boost::mpl::not_< is_accessor< Accessor > >,
                                                    is_global_accessor< Accessor > >,
            typename accessor_return_type< Accessor >::type >::type
        operator()(Accessor const &accessor) {
            GRIDTOOLS_STATIC_ASSERT(
                (is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");
            GRIDTOOLS_STATIC_ASSERT(
                (Accessor::n_dimensions > 2), "Accessor with less than 3 dimensions. Did you forget a \"!\"?");

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
        GT_FUNCTION auto operator()(Expression< Argument, exponent > const &arg)
            -> decltype(expressions::evaluation::value((*this), arg)) {

            GRIDTOOLS_STATIC_ASSERT((is_expr< Expression< Argument, exponent > >::value), "invalid expression");
            return expressions::evaluation::value((*this), arg);
        }
    };

    //    ################## IMPLEMENTATION ##############################

    /**@brief returns the value of the memory at the given address, plus the offset specified by the arg placeholder
       \param accessor Accessor pass to the evaluator
       \param storage_pointer pointer to the first element of the specific data field used
       \tparam DirectGMemAccess selects a direct access to main memory for the given accessor, ignoring if the
           parameter is being cached using software managed cache syntax
    */
    template < typename IterateDomainImpl >
    template < typename Accessor, typename StoragePointer, bool DirectGMemAccess >
    GT_FUNCTION typename iterate_domain< IterateDomainImpl >::template accessor_return_type< Accessor >::type
    iterate_domain< IterateDomainImpl >::get_value(
        Accessor const &accessor, StoragePointer const &RESTRICT storage_pointer) const {

        typedef typename iterate_domain< IterateDomainImpl >::template accessor_return_type< Accessor >::type return_t;

        // getting information about the storage
        typedef typename Accessor::index_t index_t;
        typedef typename local_domain_t::template get_arg< index_t >::type arg_t;

        typedef typename storage_wrapper_elem< arg_t, typename local_domain_t::storage_wrapper_list_t >::type
            storage_wrapper_t;
        typedef typename storage_wrapper_t::storage_info_t storage_info_t;
        typedef typename storage_wrapper_t::data_t data_t;

        // this index here describes the position of the storage info in the m_index array (can be different to the
        // storage info id)
        typedef typename boost::mpl::find< typename local_domain_t::storage_info_ptr_list,
            const storage_info_t * >::type::pos storage_info_index_t;

        const storage_info_t *storage_info =
            boost::fusion::at< storage_info_index_t >(local_domain.m_local_storage_info_ptrs);

        GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), "Using EVAL is only allowed for an accessor type");

#ifdef CUDA8
        assert(storage_pointer);
#endif
        data_t *RESTRICT real_storage_pointer = static_cast< data_t * >(storage_pointer);
#ifdef CUDA8
        assert(real_storage_pointer);
#endif

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

        if (DirectGMemAccess) {
            return get_gmem_value< return_t >(real_storage_pointer, pointer_offset);
        } else {
            return static_cast< const IterateDomainImpl * >(this)
                ->template get_value_impl<
                    typename iterate_domain< IterateDomainImpl >::template accessor_return_type< Accessor >::type,
                    Accessor,
                    data_t * >(real_storage_pointer, pointer_offset);
        }
    }

} // namespace gridtools

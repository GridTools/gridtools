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

#ifndef CXX11_ENABLED
#include <boost/typeof/typeof.hpp>
#endif
#include <boost/fusion/include/size.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/modulus.hpp>
#include <boost/mpl/for_each.hpp>
#ifdef CXX11_ENABLED
#include "expressions/expressions.hpp"
#endif
#include "../common/meta_array.hpp"
#include "../common/array.hpp"
#include "common/generic_metafunctions/static_if.hpp"
#include "common/generic_metafunctions/reversed_range.hpp"
#include "stencil-composition/total_storages.hpp"
#include "array_tuple.hpp"

/**
   @file
   @brief file implementing helper functions which are used in iterate_domain to assign/increment strides, access
   indices and storage pointers.

   All the helper functions use template recursion to implement loop unrolling
*/

namespace gridtools {

    /**
     * @brief metafunction that determines if a type is one of the storage types allowed by the iterate domain
     */
    template < typename T >
    struct is_any_iterate_domain_storage : is_storage< T > {};

    template < typename T >
    struct is_any_iterate_domain_meta_storage : is_meta_storage< T > {};

    /**
     * @brief metafunction that determines if a type is one of the storage types allowed by the iterate domain
     */
    template < typename T >
    struct is_any_iterate_domain_storage_pointer
        : boost::mpl::and_< is_any_iterate_domain_storage< typename T::value_type >, is_pointer< T > > {};

    template < typename T >
    struct is_any_iterate_domain_meta_storage_pointer
        : boost::mpl::and_< is_any_iterate_domain_meta_storage< typename T::value_type >, is_pointer< T > > {};

    /**@brief functor assigning the 'raw' data pointers to an input data pointers array (i.e. the m_data_pointers
       array).

       The 'raw' datas are the one or more data fields contained in each of the storage classes used by the current user
       function.
       @tparam Offset an index identifying the starting position in the data pointers array of the portion corresponding
       to the given storage
       @tparam BackendType the type of backend
       @tparam StrategyType the type of strategy
       @tparam DataPointerArray gridtools array of data pointers
       @tparam Storage any of the storage type handled by the iterate domain
       @tparam PEBlockSize the processing elements block size
       To clarify the meaning of the two template indices, supposing that we have a 'rectangular' vector field, NxM,
       where N is the constant number of
       snapshots per storage, while M is the number of storages. Then 'Number' would be an index between 0 and N, while
       Offset would have the form n*M, where
       0<n<N is the index of the previous storage.
    */
    template < uint_t Offset,
        typename BackendType,
        typename DataPointerArray,
        typename StoragePtr,
        typename PEBlockSize >
    struct assign_raw_data_functor {
        GRIDTOOLS_STATIC_ASSERT((is_array< DataPointerArray >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_pointer< StoragePtr >::value), "You are using an unsupported storage type ");
        GRIDTOOLS_STATIC_ASSERT((is_block_size< PEBlockSize >::value), "Error: wrong type");
        typedef typename StoragePtr::value_type storage_type;
#ifdef PEDANTIC
        GRIDTOOLS_STATIC_ASSERT((is_any_iterate_domain_storage< storage_type >::value),
            "If you are using generic accessors disable the pedantic mode. \n\
If you are not using generic accessors then you are using an unsupported storage type ");
#endif

      private:
        DataPointerArray &RESTRICT m_data_pointer_array;
        pointer< storage_type > m_storage;
        const uint_t m_offset;

      public:
        GT_FUNCTION
        assign_raw_data_functor(assign_raw_data_functor const &other)
            : m_data_pointer_array(other.m_data_pointer_array), m_storage(other.m_storage), m_offset(other.m_offset) {}

        GT_FUNCTION
        assign_raw_data_functor(
            DataPointerArray &RESTRICT data_pointer_array, pointer< storage_type > storage, uint_t const offset_)
            : m_data_pointer_array(data_pointer_array), m_storage(storage), m_offset(offset_) {}

        template < typename ID >
        GT_FUNCTION void operator()(ID const &) const {
            assert(m_storage.get());
            // provide the implementation that performs the assignment, depending on the type of storage we have
            impl< ID, storage_type >();
        }

      private:
        assign_raw_data_functor();

        // implementation of the assignment of the data pointer in case the storage is a temporary storage
        template < typename ID, typename Storage_ >
        GT_FUNCTION void impl(
            typename boost::enable_if_c< is_any_storage< Storage_ >::type::value >::type *t = 0) const {
            // TODO Add assert for m_storage->template access_value<ID>()
            // BackendType::template once_per_block< ID::value, PEBlockSize >::assign(
            //     m_data_pointer_array[Offset + ID::value], m_storage->template access_value< ID >() + m_offset);
            m_data_pointer_array[Offset + ID::value] =
                static_cast< typename Storage_::value_type * >(m_storage->template access_value< ID >() + m_offset);
        }

        /**@brief implementation in case of a generic accessor*/
        template < typename ID, typename Storage_ >
        GT_FUNCTION void impl(
            typename boost::enable_if_c< boost::mpl::not_< typename is_any_storage< Storage_ >::type >::value >::type
                *t = 0) const {
            // TODO Add assert for m_storage->template access_value<ID>()
            // BackendType::template once_per_block< ID::value, PEBlockSize >::assign(
            //     m_data_pointer_array[Offset + ID::value], m_storage->template access_value< ID >());
            m_data_pointer_array[Offset + ID::value] =
                static_cast< typename Storage_::value_type * >(m_storage->template access_value< ID >());
        }
    };

    /**@brief incrementing all the storage pointers to the m_data_pointers array

       @tparam Coordinate direction along which the increment takes place
       @tparam Execution policy determining how the increment is done (e.g. increment/decrement)
       @tparam StridesCached strides cached type
       @tparam StorageSequence sequence of storages

           This method is responsible of incrementing the index for the memory access at
           the location (i,j,k) incremented/decremented by 1 along the 'Coordinate' direction. Such index is shared
       among all the fields contained in the
           same storage class instance, and it is not shared among different storage instances.

           The actual increment computation is delegated to the storage classes, the reason being that the
       implementation may depend on the storage type
           (e.g. whether the storage is temporary, partiitoned into blocks, ...)
    */
    template < uint_t Coordinate, typename StridesCached, typename MetaStorageSequence, typename ArrayIndex >
    struct increment_index_functor {

        GRIDTOOLS_STATIC_ASSERT((is_array_tuple< StridesCached >::value), "internal error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_array_of< ArrayIndex, int >::value), "internal error: wrong type");
        GRIDTOOLS_STATIC_ASSERT(
            (is_sequence_of< MetaStorageSequence, is_pointer >::value), "internal error: wrong type");

        GT_FUNCTION
        increment_index_functor(
            int_t const increment, ArrayIndex &RESTRICT index_array, StridesCached const &RESTRICT array_tuple)
            : m_increment(increment), m_index_array(index_array), m_array_tuple(array_tuple) {}

        template < typename Pair >
        GT_FUNCTION void operator()(Pair const &) const {

            typedef typename boost::mpl::second< Pair >::type ID;
            typedef typename boost::mpl::first< Pair >::type metadata_t;

            GRIDTOOLS_STATIC_ASSERT((ID::value < boost::fusion::result_of::size< MetaStorageSequence >::value),
                "Accessing an index out of bound in fusion tuple");
            boost::mpl::at< MetaStorageSequence, ID >::type::value_type::template increment< Coordinate >(
                m_increment, &m_index_array[ID::value], m_array_tuple.template get< ID::value >());
        }

        GT_FUNCTION
        increment_index_functor(increment_index_functor const &other)
            : m_increment(other.m_increment), m_index_array(other.m_index_array), m_array_tuple(other.m_array_tuple){};

      private:
        increment_index_functor();

        const int_t m_increment;
        ArrayIndex &RESTRICT m_index_array;
        StridesCached const &RESTRICT m_array_tuple;
    };

    /**@brief assigning all the storage pointers to the m_data_pointers array

       similar to the increment_index class, but assigns the indices, and it does not depend on the storage type
    */
    template < uint_t ID >
    struct set_index_recur {
        /**@brief does the actual assignment
           This method is responsible of assigning the index for the memory access at
           the location (i,j,k). Such index is shared among all the fields contained in the
           same storage class instance, and it is not shared among different storage instances.

           This method given an array and an integer id assigns to the current component of the array the input integer.
        */
        template < typename Array >
        GT_FUNCTION static void set(int_t const &id, Array &index) {
            GRIDTOOLS_STATIC_ASSERT((is_array< Array >::value), "type is not a gridtools array");
            index[ID] = id;
            set_index_recur< ID - 1 >::set(id, index);
        }

        /**@brief does the actual assignment
           This method is responsible of assigning the index for the memory access at
           the location (i,j,k). Such index is shared among all the fields contained in the
           same storage class instance, and it is not shared among different storage instances.

           This method given two arrays copies the IDth component of one into the other, i.e. recursively cpoies one
           array into the other.
        */
        template < typename Array >
        GT_FUNCTION static void set(Array const &index, Array &out) {
            GRIDTOOLS_STATIC_ASSERT((is_array< Array >::value), "type is not a gridtools array");
            out[ID] = index[ID];
            set_index_recur< ID - 1 >::set(index, out);
        }

      private:
        set_index_recur();
        set_index_recur(set_index_recur const &);
    };

    /**usual specialization to stop the recursion*/
    template <>
    struct set_index_recur< 0 > {

        template < typename Array >
        GT_FUNCTION static void set(int_t const &id, Array &index /* , ushort_t* lru */) {
            GRIDTOOLS_STATIC_ASSERT((is_array< Array >::value), "type is not a gridtools array");
            index[0] = id;
        }

        template < typename Array >
        GT_FUNCTION static void set(Array const &index, Array &out) {
            GRIDTOOLS_STATIC_ASSERT((is_array< Array >::value), "type is not a gridtools array");
            out[0] = index[0];
        }
    };

    /**@brief functor initializing the indeces
     *     does the actual assignment
     *     This method is responsible of computing the index for the memory access at
     *     the location (i,j,k). Such index is shared among all the fields contained in the
     *     same storage class instance, and it is not shared among different storage instances.
     * @tparam Coordinate direction along which the increment takes place
     * @tparam StridesCached strides cached type
     * @tparam StorageSequence sequence of storages
     */
    template < uint_t Coordinate, typename Strides, typename MetaStorageSequence, typename ArrayIndex >
    struct initialize_index_functor {
      private:
        GRIDTOOLS_STATIC_ASSERT((is_array_tuple< Strides >::value), "internal error: wrong type");
        GRIDTOOLS_STATIC_ASSERT(
            (is_sequence_of< MetaStorageSequence, is_pointer >::value), "internal error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_array_of< ArrayIndex, int >::value), "internal error: wrong type");

        Strides const &RESTRICT m_strides;
        const int_t m_initial_pos;
        const uint_t m_block;
        array< uint_t, 3 > const m_initial_offsets;

        ArrayIndex &RESTRICT m_index_array;
        initialize_index_functor();

      public:
        GT_FUNCTION
        initialize_index_functor(initialize_index_functor const &other, array< uint_t, 3 > const &initial_offsets_)
            : m_strides(other.m_strides), m_initial_pos(other.m_initial_pos), m_block(other.m_block),
              m_index_array(other.m_index_array), m_initial_offsets(initial_offsets_) {}

        GT_FUNCTION
        initialize_index_functor(Strides const &RESTRICT strides
            // MetaStorageSequence const &RESTRICT storages,
            ,
            const int_t initial_pos,
            const uint_t block,
            ArrayIndex &RESTRICT index_array,
            array< uint_t, 3 > const &initial_offsets_)
            : m_strides(strides), m_initial_pos(initial_pos), m_block(block), m_index_array(index_array),
              m_initial_offsets(initial_offsets_) {}

        template < typename Pair >
        GT_FUNCTION void operator()(Pair const &) const {

            typedef typename boost::mpl::second< Pair >::type id_t;
            GRIDTOOLS_STATIC_ASSERT((id_t::value < boost::fusion::result_of::size< MetaStorageSequence >::value),
                "Accessing an index out of bound in fusion tuple");

            boost::mpl::at< MetaStorageSequence, id_t >::type::value_type::template initialize< Coordinate >(
                m_initial_pos,
                m_block,
                &m_index_array[id_t::value],
                m_strides.template get< id_t::value >(),
                m_initial_offsets);
        }
    };

    /**@brief functor assigning all the storage pointers to the m_data_pointers array
     * This method is responsible of copying the base pointers of the storages inside a local vector
     * which is tipically instantiated on a fast local memory.
     *
     * The EU stands for ExecutionUnit (thich may be a thread or a group of
     * threads. There are potentially two ids, one over i and one over j, since
     * our execution model is parallel on (i,j). Defaulted to 1.
     * @tparam BackendType the type of backend
     * @tparam DataPointerArray gridtools array of data pointers
     * @tparam StorageSequence sequence of any of the storage types handled by the iterate domain
     * @tparam PEBlockSize the processing elements block size
     * */
    template < typename BackendType,
        typename DataPointerArray,
        typename StorageSequence,
        typename MetaStorageSequence,
        typename MetaDataMap,
        typename PEBlockSize >
    struct assign_storage_functor {

        GRIDTOOLS_STATIC_ASSERT((is_array< DataPointerArray >::value), "internal error: wrong type");

        GRIDTOOLS_STATIC_ASSERT(
            (is_sequence_of< StorageSequence, is_pointer >::value), "You are using an unsupported storage type ");
        GRIDTOOLS_STATIC_ASSERT((is_block_size< PEBlockSize >::value), "Error: wrong type");

#ifdef PEDANTIC
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< StorageSequence, is_any_iterate_domain_storage_pointer >::value),
            "If you are using generic accessors disable the pedantic mode. \n If you are not using generic accessors "
            "then you are using an unsupported storage type ");
#endif

      private:
        DataPointerArray &RESTRICT m_data_pointer_array;
        StorageSequence const &RESTRICT m_storages;
        MetaStorageSequence const &RESTRICT m_meta_storages;
        const int_t m_EU_id_i;
        const int_t m_EU_id_j;
        assign_storage_functor();

      public:
        GT_FUNCTION
        assign_storage_functor(assign_storage_functor const &other)
            : m_data_pointer_array(other.m_data_pointer_array), m_storages(other.m_storages),
              m_meta_storages(other.m_meta_storages), m_EU_id_i(other.m_EU_id_i), m_EU_id_j(other.m_EU_id_j) {}

        GT_FUNCTION
        assign_storage_functor(DataPointerArray &RESTRICT data_pointer_array,
            StorageSequence const &RESTRICT storages,
            MetaStorageSequence const &RESTRICT meta_storages,
            const int_t EU_id_i,
            const int_t EU_id_j)
            : m_data_pointer_array(data_pointer_array), m_storages(storages), m_meta_storages(meta_storages),
              m_EU_id_i(EU_id_i), m_EU_id_j(EU_id_j) {}

        /**Metafunction used in the enable_if below*/
        template < typename ID >
        struct any_supported_accessor_t {
            typedef is_any_storage< typename boost::mpl::at< StorageSequence, ID >::type > type;
        };

        /**
           @brief Overload when the accessor associated with this ID is not a user-defined global accessor

         */
        template < typename ID >
        GT_FUNCTION void operator()(ID const &,
            typename boost::enable_if< typename any_supported_accessor_t< ID >::type, int >::type dummy = 0) const {
            GRIDTOOLS_STATIC_ASSERT((ID::value < boost::fusion::result_of::size< StorageSequence >::value),
                "Accessing an index out of bound in fusion tuple");

            typedef typename boost::mpl::at< StorageSequence, ID >::type storage_ptr_type;
            typedef typename storage_ptr_type::value_type storage_type;

            typedef
                typename boost::mpl::at< MetaDataMap, typename storage_type::storage_info_type >::type metadata_index_t;

            pointer< const typename storage_type::storage_info_type > const metadata_ =
                boost::fusion::at< metadata_index_t >(m_meta_storages);

            // if the following fails, the ID is larger than the number of storage types
            GRIDTOOLS_STATIC_ASSERT(ID::value < boost::mpl::size< StorageSequence >::value,
                "the ID is larger than the number of storage types");

            boost::mpl::for_each< typename reversed_range< ushort_t, 0, storage_type::field_dimensions >::type >(
                assign_raw_data_functor< total_storages< StorageSequence, ID::value >::value,
                    BackendType,
                    DataPointerArray,
                    storage_ptr_type,
                    PEBlockSize >(m_data_pointer_array,
                    boost::fusion::at< ID >(m_storages),
                    metadata_->fields_offset(m_EU_id_i, m_EU_id_j)));
        }

        /**
           @brief Overload when the accessor associated with this ID is a user-defined global accessor

           assigns the storage pointers in the iterate_domain
         */
        template < typename ID >
        GT_FUNCTION void operator()(ID const &,
            typename boost::disable_if< typename any_supported_accessor_t< ID >::type, int >::type dummy = 0) const {
            GRIDTOOLS_STATIC_ASSERT((ID::value < boost::fusion::result_of::size< StorageSequence >::value),
                "Accessing an index out of bound in fusion tuple");

            typedef typename boost::remove_reference< typename boost::mpl::at< StorageSequence, ID >::type >::type
                storage_ptr_type;
            typedef typename storage_ptr_type::value_type storage_type;

            // if the following fails, the ID is larger than the number of storage types
            GRIDTOOLS_STATIC_ASSERT(ID::value < boost::mpl::size< StorageSequence >::value,
                "the ID is larger than the number of storage types");

            boost::mpl::for_each< typename reversed_range< ushort_t, 0, storage_type::field_dimensions >::type >(
                assign_raw_data_functor< total_storages< StorageSequence, ID::value >::value,
                    BackendType,
                    DataPointerArray,
                    storage_ptr_type,
                    PEBlockSize >(
                    m_data_pointer_array, boost::fusion::at< ID >(m_storages), 0u /* hardcoded offset */));
        }
    };

    /**
     * metafunction that evaluates if an accessor is cached by the backend
     * the Accessor parameter is either an Accessor or an expressions
     */
    template < typename Accessor, typename CachesMap >
    struct accessor_is_cached {
        template < typename Accessor_ >
        struct accessor_is_cached_ {
            GRIDTOOLS_STATIC_ASSERT((is_accessor< Accessor >::value), "Error: wrong type");
            typedef typename boost::mpl::has_key< CachesMap, typename accessor_index< Accessor_ >::type >::type type;
        };

        typedef typename boost::mpl::eval_if< is_accessor< Accessor >,
            accessor_is_cached_< Accessor >,
            boost::mpl::identity< boost::mpl::false_ > >::type type;

        BOOST_STATIC_CONSTANT(bool, value = (type::value));
    };

    template < typename LocalDomain, typename Accessor >
    struct get_arg_accessor {
        GRIDTOOLS_STATIC_ASSERT(is_local_domain< LocalDomain >::value, "Wrong type");
        GRIDTOOLS_STATIC_ASSERT(is_accessor< Accessor >::value, "Wrong type");

        typedef typename LocalDomain::domain_args_t arg_list_t;

        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< arg_list_t >::value > Accessor::index_type::value),
            "accessor has an ID which is too large");

        typedef typename boost::mpl::at< arg_list_t, typename Accessor::index_type >::type type;
    };

    template < typename LocalDomain, typename Accessor >
    struct get_storage_accessor {
        typedef typename get_arg_accessor< LocalDomain, Accessor >::type arg_type;

        typedef typename arg_type::storage_type type;
    };
    template < typename LocalDomain, typename Accessor >
    struct get_storage_pointer_accessor {
        GRIDTOOLS_STATIC_ASSERT(is_local_domain< LocalDomain >::value, "Wrong type");
        GRIDTOOLS_STATIC_ASSERT(is_accessor< Accessor >::value, "Wrong type");

        GRIDTOOLS_STATIC_ASSERT(
            (boost::mpl::size< typename LocalDomain::domain_args_t >::value > Accessor::index_type::value),
            "Wrong type");

        typedef typename boost::add_pointer<
            typename get_storage_accessor< LocalDomain, Accessor >::type::value_type >::type type;
    };

    template < typename T >
    struct get_storage_type {
        typedef T type;
    };

    template < typename T >
    struct get_storage_type< std::vector< pointer< T > > > {
        typedef T type;
    };

    /**
     * metafunction that retrieves the arg type associated with an accessor
     */
    template < typename Accessor, typename IterateDomainArguments >
    struct get_arg_from_accessor {
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments< IterateDomainArguments >::value), "Wrong type");

        typedef typename boost::mpl::at< typename IterateDomainArguments::local_domain_t::esf_args,
            typename Accessor::index_type >::type type;
    };

    template < typename Accessor, typename IterateDomainArguments >
    struct get_arg_value_type_from_accessor {
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments< IterateDomainArguments >::value), "Wrong type");

        GRIDTOOLS_STATIC_ASSERT(
            !(boost::is_same< typename get_arg_from_accessor< Accessor, IterateDomainArguments >::type,
                boost::mpl::void_ >::type::value),
            "No argument matching a given accessor. You probably forgot one argument in the call to a stage of the "
            "computation.");
        typedef typename get_storage_type< typename get_arg_from_accessor< Accessor,
            IterateDomainArguments >::type::storage_type >::type::value_type type;
    };

    /**
       @brief partial specialization for the global_accessor

       for the global accessor the value_type is the storage object type itself.
    */
    template < int_t I, enumtype::intend Intend, typename IterateDomainArguments >
    struct get_arg_value_type_from_accessor< global_accessor< I, Intend >, IterateDomainArguments > {
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments< IterateDomainArguments >::value), "Wrong type");

        typedef typename boost::mpl::at< typename IterateDomainArguments::local_domain_t::mpl_storages,
            static_int< I > >::type::value_type::value_type type;
    };

    /**
     * metafunction that computes the return type of all operator() of an accessor
     */
    template < typename Accessor, typename IterateDomainArguments >
    struct accessor_return_type_impl {
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments< IterateDomainArguments >::value), "Wrong type");
        typedef typename boost::remove_reference< Accessor >::type acc_t;

        typedef typename boost::mpl::eval_if< boost::mpl::or_< is_accessor< acc_t >, is_vector_accessor< acc_t > >,
            get_arg_value_type_from_accessor< acc_t, IterateDomainArguments >,
            boost::mpl::identity< boost::mpl::void_ > >::type accessor_value_type;

        typedef typename boost::mpl::if_< is_accessor_readonly< acc_t >,
            typename boost::add_const< accessor_value_type >::type,
            typename boost::add_reference< accessor_value_type >::type RESTRICT >::type type;
    };

} // namespace gridtools

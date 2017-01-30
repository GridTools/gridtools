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
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/modulus.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/utility/enable_if.hpp>
#ifdef CXX11_ENABLED
#include "expressions/expressions.hpp"
#endif
#include "../common/array.hpp"
#include "../common/meta_array.hpp"
#include "common/generic_metafunctions/reversed_range.hpp"
#include "common/generic_metafunctions/static_if.hpp"
#include "stencil-composition/total_storages.hpp"

/**
   @file
   @brief file implementing helper functions which are used in iterate_domain to assign/increment strides, access
   indices and storage pointers.

   All the helper functions use template recursion to implement loop unrolling
*/

namespace gridtools {

    namespace {
        template < class InputIt, class OutputIt >
        GT_FUNCTION OutputIt copy_ptrs(InputIt first, InputIt last, OutputIt d_first, const int offset = 0) {
            while (first != last) {
                *d_first++ = (*first++) + offset;
            }
            return d_first;
        }
    }

    /* data structure that can be used to store the data pointers of a given list of storages */
    template < typename StorageWrapperList, int I = boost::mpl::size< StorageWrapperList >::value - 1 >
    struct data_ptr_cached : data_ptr_cached< StorageWrapperList, I - 1 > {
        typedef data_ptr_cached< StorageWrapperList, I - 1 > super;
        typedef typename boost::mpl::at_c< StorageWrapperList, I >::type storage_wrapper_t;
        typedef void *data_ptr_t[storage_wrapper_t::storage_size];

        constexpr static int index = I;

        data_ptr_t m_content;

        template < short_t Idx >
        using return_t = typename boost::mpl::if_< boost::mpl::bool_< Idx == index >,
            data_ptr_t,
            typename super::template return_t< Idx > >::type;

        template < short_t Idx >
        GT_FUNCTION return_t< Idx > const &RESTRICT get() const {
            return static_if< (Idx == index) >::apply(m_content, super::template get< Idx >());
        }

        template < short_t Idx >
        GT_FUNCTION return_t< Idx > &RESTRICT get() {
            return static_if< (Idx == index) >::apply(m_content, super::template get< Idx >());
        }
    };

    template < typename StorageWrapperList >
    struct data_ptr_cached< StorageWrapperList, 0 > {
        typedef typename boost::mpl::at_c< StorageWrapperList, 0 >::type storage_wrapper_t;
        typedef void *data_ptr_t[storage_wrapper_t::storage_size];

        constexpr static int index = 0;

        data_ptr_t m_content;

        template < short_t Idx >
        using return_t = data_ptr_t;

        template < short_t Idx >
        GT_FUNCTION data_ptr_t const &RESTRICT get() const {
            return m_content;
        }

        template < short_t Idx >
        GT_FUNCTION data_ptr_t &RESTRICT get() {
            return m_content;
        }
    };

    template < typename T >
    struct is_data_ptr_cached : boost::mpl::false_ {};

    template < typename StorageWrapperList, int ID >
    struct is_data_ptr_cached< data_ptr_cached< StorageWrapperList, ID > > : boost::mpl::true_ {};

    /**
       @brief struct to allocate recursively all the strides with the proper dimension

       the purpose of this struct is to allocate the storage for the strides of a set of storages. Tipically
       it is used to cache these strides in a fast memory (e.g. shared memory).
       \tparam ID recursion index, representing the current storage
       \tparam StorageInfoList typelist of the storages
    */
    template < ushort_t ID, typename StorageInfoList >
    struct strides_cached : public strides_cached< ID - 1, StorageInfoList > {
        GRIDTOOLS_STATIC_ASSERT(boost::mpl::size< StorageInfoList >::value > ID,
            "Library internal error: strides index exceeds the number of storages");
        typedef typename boost::mpl::at_c< StorageInfoList, ID >::type storage_info_ptr_t;
        typedef typename boost::remove_pointer< typename boost::remove_cv< storage_info_ptr_t >::type >::type
            storage_info_t;
        typedef strides_cached< ID - 1, StorageInfoList > super;
        typedef array< int_t, storage_info_t::Layout::length - 1 > data_array_t;

        template < short_t Idx >
        using return_t = typename boost::mpl::if_< boost::mpl::bool_< Idx == ID >,
            data_array_t,
            typename super::template return_t< Idx > >::type;

        /**@brief constructor, doing nothing more than allocating the space*/
        GT_FUNCTION
        strides_cached() : super() {}

        template < short_t Idx >
        GT_FUNCTION return_t< Idx > const &RESTRICT get() const {
            return static_if< (Idx == ID) >::apply(m_data, super::template get< Idx >());
        }

        template < short_t Idx >
        GT_FUNCTION return_t< Idx > &RESTRICT get() {
            return static_if< (Idx == ID) >::apply(m_data, super::template get< Idx >());
        }

      private:
        data_array_t m_data;
        strides_cached(strides_cached const &);
    };

    /**specialization to stop the recursion*/
    template < typename StorageInfoList >
    struct strides_cached< (ushort_t)0, StorageInfoList > {
        typedef typename boost::mpl::at_c< StorageInfoList, 0 >::type storage_info_ptr_t;
        typedef typename boost::remove_pointer< typename boost::remove_cv< storage_info_ptr_t >::type >::type
            storage_info_t;

        GT_FUNCTION
        strides_cached() {}

        typedef array< int_t, storage_info_t::Layout::length - 1 > data_array_t;

        template < short_t Idx >
        using return_t = data_array_t;

        template < short_t Idx >
        GT_FUNCTION data_array_t &RESTRICT get() { // stop recursion
            return m_data;
        }

        template < short_t Idx >
        GT_FUNCTION data_array_t const &RESTRICT get() const { // stop recursion
            return m_data;
        }

      private:
        data_array_t m_data;
        strides_cached(strides_cached const &);
    };

    template < typename T >
    struct is_strides_cached : boost::mpl::false_ {};

    template < uint_t ID, typename StorageInfoList >
    struct is_strides_cached< strides_cached< ID, StorageInfoList > > : boost::mpl::true_ {};

    /**@brief incrementing all the storage pointers to the m_data_pointers array

       @tparam Coordinate direction along which the increment takes place
       @tparam Execution policy determining how the increment is done (e.g. increment/decrement)
       @tparam StridesCached strides cached type

           This method is responsible of incrementing the index for the memory access at
           the location (i,j,k) incremented/decremented by 1 along the 'Coordinate' direction. Such index is shared
           among all the fields contained in the
           same storage class instance, and it is not shared among different storage instances.
    */
    template < typename LocalDomain, uint_t Coordinate, typename StridesCached, typename ArrayIndex >
    struct increment_index_functor {
        GRIDTOOLS_STATIC_ASSERT((is_strides_cached< StridesCached >::value), "internal error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_array_of< ArrayIndex, int >::value), "internal error: wrong type");

        const int_t m_increment;
        ArrayIndex &RESTRICT m_index_array;
        StridesCached &RESTRICT m_strides_cached;

        GT_FUNCTION
        increment_index_functor(
            int_t const increment, ArrayIndex &RESTRICT index_array, StridesCached &RESTRICT strides_cached)
            : m_increment(increment), m_index_array(index_array), m_strides_cached(strides_cached) {}

        template < typename StorageInfo >
        GT_FUNCTION void operator()(const StorageInfo *sinfo) const {
            typedef typename boost::mpl::find< typename LocalDomain::storage_info_ptr_list,
                const StorageInfo * >::type::pos index_t;
            typedef
                typename boost::mpl::at< typename LocalDomain::storage_info_tmp_info_t, StorageInfo >::type tmp_info_t;

            GRIDTOOLS_STATIC_ASSERT(
                (index_t::value < ArrayIndex::n_dimensions), "Accessing an index out of bound in fusion tuple");
            
            // get the max coordinate of given StorageInfo
            typedef typename boost::mpl::deref< typename boost::mpl::max_element<
                typename StorageInfo::Layout::static_layout_vector >::type >::type max_t;

            // get the position
            constexpr int pos = StorageInfo::Layout::template at< Coordinate >();
            if (pos >= 0) {
                auto stride = (max_t::value < 0)
                                  ? 0
                                  : ((pos == max_t::value) ? 1 : m_strides_cached.template get< index_t::value >()[pos]);
                m_index_array[index_t::value] += (stride * m_increment);
            }
        }

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
    template < uint_t Coordinate, typename Strides, typename LocalDomain, typename ArrayIndex, typename PEBlockSize >
    struct initialize_index_functor {
      private:
        GRIDTOOLS_STATIC_ASSERT((is_strides_cached< Strides >::value), "internal error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_array_of< ArrayIndex, int >::value), "internal error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_block_size< PEBlockSize >::value), "Error: wrong type");

        Strides &RESTRICT m_strides;
        const int_t m_initial_pos;
        const uint_t m_block;
        ArrayIndex &RESTRICT m_index_array;
        initialize_index_functor();

      public:
        GT_FUNCTION
        initialize_index_functor(initialize_index_functor const &other)
            : m_strides(other.m_strides), m_initial_pos(other.m_initial_pos), m_block(other.m_block),
              m_index_array(other.m_index_array) {}

        GT_FUNCTION
        initialize_index_functor(
            Strides &RESTRICT strides, const int_t initial_pos, const uint_t block, ArrayIndex &RESTRICT index_array)
            : m_strides(strides), m_initial_pos(initial_pos), m_block(block), m_index_array(index_array) {}

        template < typename StorageInfo >
        GT_FUNCTION void operator()(const StorageInfo *storage_info) const {
            typedef typename boost::mpl::find< typename LocalDomain::storage_info_ptr_list,
                const StorageInfo * >::type::pos index_t;
            // check if the current storage info a temporary
            typedef
                typename boost::mpl::at< typename LocalDomain::storage_info_tmp_info_t, StorageInfo >::type tmp_info_t;
            // get the max coordinate of given StorageInfo
            typedef typename boost::mpl::deref< typename boost::mpl::max_element<
                typename StorageInfo::Layout::static_layout_vector >::type >::type max_t;

            GRIDTOOLS_STATIC_ASSERT(
                (index_t::value < ArrayIndex::n_dimensions), "Accessing an index out of bound in fusion tuple");
            const int_t initial_pos = (tmp_info_t::value) ? 
                ((m_initial_pos)-m_block * ((Coordinate == 1) ? PEBlockSize::j_size_t::value : ((Coordinate == 0) ? PEBlockSize::i_size_t::value : 0))) :
                m_initial_pos;
            constexpr int pos = StorageInfo::Layout::template at< Coordinate >();
            if (Coordinate < StorageInfo::Layout::length && pos >= 0) {
                auto stride = (max_t::value < 0)
                                  ? 0
                                  : ((pos == max_t::value) ? 1 : m_strides.template get< index_t::value >()[pos]);
                m_index_array[index_t::value] += (stride * initial_pos);
            }            
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
     * @tparam LocalDomain local domain type
     * @tparam PEBlockSize the processing elements block size
     * */
    template < typename Backend,
        typename DataPtrCached,
        typename LocalDomain,
        typename PEBlockSize,
        typename ExtentMap,
        typename MaxExtents >
    struct assign_storage_ptrs {

        GRIDTOOLS_STATIC_ASSERT((is_data_ptr_cached< DataPtrCached >::value), "Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_block_size< PEBlockSize >::value), "Error: wrong type");
        typedef typename LocalDomain::storage_info_ptr_fusion_list storage_info_ptrs_t;

        DataPtrCached RESTRICT &m_data_ptr_cached;
        storage_info_ptrs_t const RESTRICT &m_storageinfo_fusion_list;

        GT_FUNCTION assign_storage_ptrs(
            DataPtrCached RESTRICT &data_ptr_cached, storage_info_ptrs_t const RESTRICT &storageinfo_fusion_list)
            : m_data_ptr_cached(data_ptr_cached), m_storageinfo_fusion_list(storageinfo_fusion_list) {}

        template < typename FusionPair >
        GT_FUNCTION void operator()(FusionPair const &sw) const {
            typedef typename boost::fusion::result_of::first< FusionPair >::type arg_t;
            typedef typename get_storage_wrapper_elem< arg_t, typename LocalDomain::storage_wrapper_list_t >::type
                storage_wrapper_t;
            typedef typename boost::mpl::find< typename LocalDomain::storage_wrapper_list_t,
                storage_wrapper_t >::type::pos pos_in_storage_wrapper_list_t;

            typedef typename boost::mpl::find< typename LocalDomain::storage_info_ptr_list,
                const typename storage_wrapper_t::storage_info_t * >::type::pos si_index_t;
            typedef typename boost::mpl::at< ExtentMap, arg_t >::type max_extent_t;

            const int offset = Backend::template fields_offset< LocalDomain,
                PEBlockSize,
                storage_wrapper_t::is_temporary,
                max_extent_t,
                MaxExtents >(boost::fusion::at< si_index_t >(m_storageinfo_fusion_list));
            for (unsigned i = 0; i < storage_wrapper_t::storage_size; ++i) {
                Backend::template once_per_block< pos_in_storage_wrapper_list_t::value, PEBlockSize >::assign(
                    m_data_ptr_cached.template get< pos_in_storage_wrapper_list_t::value >()[i], sw.second[i] + offset);
            }

            /*
                        printf("Assign storage ptr for arg %i\n", storage_wrapper_t::index_t::value);
                        printf("pos in storage wrapper list: %i\n", pos_in_storage_wrapper_list_t::value);
                        for(unsigned i=0; i<storage_wrapper_t::storage_size; ++i)
                            printf("\t%p\n", m_data_ptr_cached.template get<pos_in_storage_wrapper_list_t::value>()[i]);
            */
        }
    };

    /**@brief functor assigning the strides to a lobal array (i.e. m_strides).

       It implements the unrolling of a double loop: i.e. is n_f is the number of fields in this user function,
       and n_d(i) is the number of space dimensions per field (dependent on the ith field), then the loop for assigning
       the strides
       would look like
       for(i=0; i<n_f; ++i)
       for(j=0; j<n_d(i); ++j)
       * @tparam BackendType the type of backend
       * @tparam StridesCached strides cached type
       * @tparam LocalDomain local domain type
       * @tparam PEBlockSize the processing elements block size
       */
    template < typename BackendType, typename StridesCached, typename LocalDomain, typename PEBlockSize >
    struct assign_strides {
        GRIDTOOLS_STATIC_ASSERT((is_block_size< PEBlockSize >::value), "Error: wrong type");

        template < typename SInfo >
        struct assign {
            const SInfo *m_storage_info;
            StridesCached RESTRICT &m_strides_cached;

            GT_FUNCTION assign(const SInfo *storage_info, StridesCached RESTRICT &strides_cached)
                : m_storage_info(storage_info), m_strides_cached(strides_cached) {}

            template < typename Coordinate >
            GT_FUNCTION typename boost::enable_if_c< (Coordinate::value >= SInfo::Layout::unmasked_length), void >::type
            operator()(Coordinate) {}

            template < typename Coordinate >
            GT_FUNCTION typename boost::enable_if_c< (Coordinate::value < SInfo::Layout::unmasked_length), void >::type
            operator()(Coordinate) {
                typedef typename SInfo::Layout layout_map_t;
                typedef typename boost::mpl::find< typename LocalDomain::storage_info_ptr_list,
                    const SInfo * >::type::pos index_t;
                GRIDTOOLS_STATIC_ASSERT(
                    (boost::mpl::contains< typename LocalDomain::storage_info_ptr_list, const SInfo * >::value),
                    "Error when trying to assign the strides in iterate domain. Access out of bounds.");
                constexpr int pos = SInfo::Layout::template find< Coordinate::value >();
                GRIDTOOLS_STATIC_ASSERT((pos < SInfo::Layout::length),
                    "Error when trying to assign the strides in iterate domain. Access out of bounds.");
                BackendType::template once_per_block< index_t::value, PEBlockSize >::assign(
                    (m_strides_cached.template get< index_t::value >())[Coordinate::value],
                    m_storage_info->template stride< pos >());
            }
        };

        StridesCached RESTRICT &m_strides_cached;

        GT_FUNCTION assign_strides(StridesCached RESTRICT &strides_cached) : m_strides_cached(strides_cached) {}

        template < typename StorageInfo >
        GT_FUNCTION typename boost::enable_if_c< StorageInfo::Layout::unmasked_length == 0, void >::type operator()(
            const StorageInfo *storage_info) const {}

        template < typename StorageInfo >
        GT_FUNCTION typename boost::enable_if_c< StorageInfo::Layout::unmasked_length, void >::type operator()(
            const StorageInfo *storage_info) const {
            boost::mpl::for_each< boost::mpl::range_c< short_t, 0, StorageInfo::Layout::unmasked_length - 1 > >(
                assign< StorageInfo >(storage_info, m_strides_cached));
            /*
                        printf("Assign strides for storage info %i\n", StorageInfo::id);
                        for(unsigned i=0; i<StorageInfo::Layout::unmasked_length - 1; ++i)
                            printf("\t%i\n", (m_strides_cached.template get<StorageInfo::id>())[i]);
            */
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
    struct get_storage_accessor {
        GRIDTOOLS_STATIC_ASSERT(is_local_domain< LocalDomain >::value, "Wrong type");
        GRIDTOOLS_STATIC_ASSERT(is_accessor< Accessor >::value, "Wrong type");

        GRIDTOOLS_STATIC_ASSERT(
            (boost::mpl::size< typename LocalDomain::data_ptr_fusion_map >::value > Accessor::index_t::value),
            "Wrong type");
        typedef typename LocalDomain::template get_storage< Accessor::index_t >::type storage_t;
        typedef storage_t type;
    };

    template < typename LocalDomain, typename Accessor >
    struct get_storage_pointer_accessor {
        GRIDTOOLS_STATIC_ASSERT(is_local_domain< LocalDomain >::value, "Wrong type");
        GRIDTOOLS_STATIC_ASSERT(is_accessor< Accessor >::value, "Wrong type");

        GRIDTOOLS_STATIC_ASSERT(
            (boost::mpl::size< typename LocalDomain::data_ptr_fusion_map >::value > Accessor::index_t::value),
            "Wrong type");

        typedef typename boost::add_pointer<
            typename get_storage_accessor< LocalDomain, Accessor >::type::value_type::value_type >::type type;
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
            typename Accessor::index_t >::type type;
    };

    template < typename Accessor, typename IterateDomainArguments >
    struct get_arg_value_type_from_accessor {
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments< IterateDomainArguments >::value), "Wrong type");

        typedef typename get_storage_type<
            typename get_arg_from_accessor< Accessor, IterateDomainArguments >::type::storage_t >::type::data_t type;
    };

    /**
       @brief partial specialization for the global_accessor

       for the global accessor the value_type is the storage object type itself.
    */
    template < ushort_t I, enumtype::intend Intend, typename IterateDomainArguments >
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

    template < typename Max, typename StridesCached, typename OffsetTuple, typename StorageInfo, unsigned N >
    GT_FUNCTION constexpr typename boost::enable_if_c< (N < (OffsetTuple::n_dim - 1)), int_t >::type apply_accessor(
        StridesCached const &RESTRICT strides, OffsetTuple const &RESTRICT offsets) {
        typedef boost::mpl::int_< (StorageInfo::Layout::template at< N >()) > val_t;
        static_assert((val_t::value == Max::value) || (N < StorageInfo::Layout::length), "invalid stride array access");
        typedef boost::mpl::bool_< (StorageInfo::Layout::template at< N >() == Max::value) > is_max_t;
        typedef boost::mpl::bool_< (StorageInfo::Layout::template at< N >() == -1) > is_masked_t;
        return (is_max_t::value ? 1 : (is_masked_t::value ? 0 : strides[val_t::value])) *
                   offsets.template get< (OffsetTuple::n_dim - 1) - N >() +
               apply_accessor< Max, StridesCached, OffsetTuple, StorageInfo, N + 1 >(strides, offsets);
    }

    template < typename Max, typename StridesCached, typename OffsetTuple, typename StorageInfo, unsigned N >
    GT_FUNCTION constexpr typename boost::enable_if_c< (N == (OffsetTuple::n_dim - 1)), int_t >::type apply_accessor(
        StridesCached const &RESTRICT strides, OffsetTuple const &RESTRICT offsets) {
        typedef boost::mpl::int_< (StorageInfo::Layout::template at< N >()) > val_t;
        static_assert((val_t::value == Max::value) || (N < StorageInfo::Layout::length), "invalid stride array access");
        typedef boost::mpl::bool_< (val_t::value == Max::value) > is_max_t;
        typedef boost::mpl::bool_< (val_t::value == -1) > is_masked_t;
        return (is_max_t::value ? 1 : (is_masked_t::value ? 0 : strides[val_t::value])) *
               offsets.template get< (OffsetTuple::n_dim - 1) - N >();
    }

    // pointer offset computation
    template < typename StorageWrapper, typename StorageInfo, typename AccessorOffset, typename StridesCached >
    GT_FUNCTION constexpr int_t compute_offset(
        StridesCached const &RESTRICT strides_cached, AccessorOffset const &RESTRICT acc_offset) {
        // get the max coordinate of given StorageInfo
        typedef typename boost::mpl::deref<
            typename boost::mpl::max_element< typename StorageInfo::Layout::static_layout_vector >::type >::type max_t;
        return apply_accessor< max_t, StridesCached, AccessorOffset, StorageInfo, 0 >(strides_cached, acc_offset);
    };

} // namespace gridtools

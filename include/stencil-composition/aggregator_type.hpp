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

#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/container/vector/convert.hpp>
#include <boost/fusion/include/as_set.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/copy.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/push_back.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/fusion/mpl.hpp>
#include <boost/fusion/view/filter_view.hpp>
#include <boost/mpl/filter_view.hpp>
#include <boost/mpl/find.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/max_element.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/sort.hpp>
#include <boost/mpl/transform.hpp>

#include "../common/generic_metafunctions/arg_comparator.hpp"
#include "../common/generic_metafunctions/gt_integer_sequence.hpp"
#include "../common/generic_metafunctions/is_sequence_of.hpp"
#include "../common/generic_metafunctions/is_variadic_pack_of.hpp"
#include "../common/generic_metafunctions/static_if.hpp"
#include "../common/generic_metafunctions/variadic_to_vector.hpp"

#include "../common/metadata_set.hpp"
#include <storage-facility.hpp>

#include "aggregator_type_impl.hpp"
#include "arg.hpp"
#include "arg_metafunctions.hpp"

/**@file
   @brief This file contains the global list of placeholders to the storages
*/
namespace gridtools {

    /**
       @brief This struct contains the global list of placeholders to the storages
       @tparam Placeholders list of placeholders of type arg<I,T,B>

     NOTE: Note on the terminology: we call "global" the quantities having the "computation" granularity,
     and "local" the quantities having the "ESF" or "MSS" granularity. This class holds the global list
     of placeholders, i.e. all the placeholders used in the current computation. This list will be
     split into several local lists, one per ESF (or fused MSS).

     This class reorders the placeholders according to their indices, checks that there are no holes in the numeration,
     etc.
    */
    template < typename Placeholders >
    struct aggregator_type {

        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< Placeholders >::type::value > 0),
            "The aggregator_type must be constructed with at least one storage placeholder. If you don't use any "
            "storage you are probably trying to do something which is not a stencil operation, aren't you?");
        typedef typename boost::mpl::sort< Placeholders, arg_comparator >::type sorted_placeholders_t;

        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< sorted_placeholders_t, is_arg >::type::value),
            "wrong type: the aggregator_type template argument must be an MPL vector of placeholders (arg<...>)");

        const static uint_t len = boost::mpl::size< sorted_placeholders_t >::type::value;

        // create a unique id that will be used as temporary storage info id
        typedef typename boost::mpl::next<
            typename boost::mpl::deref< typename boost::mpl::max_element< typename boost::mpl::transform< Placeholders,
                _impl::extract_storage_info_id_from_arg >::type >::type >::type >::type tmp_storage_info_id_t;

        // replace the storage_info_t of all temporary args with the new index type
        typedef
            typename boost::mpl::fold< sorted_placeholders_t,
                boost::mpl::vector0<>,
                boost::mpl::push_back< boost::mpl::_1,
                                           boost::mpl::if_< is_tmp_arg< boost::mpl::_2 >,
                                               _impl::replace_arg_storage_info< tmp_storage_info_id_t, boost::mpl::_2 >,
                                               boost::mpl::_2 > > >::type placeholders_t;

        // filter out the storage infos which are the same
        typedef typename boost::mpl::fold<
            placeholders_t,
            boost::mpl::set0<>, // check if the argument is a data_store placeholder before extracting the storage_info
            boost::mpl::if_< is_storage_info< get_storage_info_from_arg< boost::mpl::_2 > >,
                boost::mpl::insert< boost::mpl::_1, get_storage_info_from_arg< boost::mpl::_2 > >,
                boost::mpl::_1 > >::type mpl_metadata_set_t;

        // create an mpl::vector of metadata types
        typedef typename boost::mpl::fold< mpl_metadata_set_t,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 > >::type metadata_vector_t;
        const static uint_t len_meta = boost::mpl::size< metadata_vector_t >::type::value;

        // Get a sequence of the same type of placeholders_t, but containing the storage types for each placeholder
        typedef typename boost::mpl::transform< placeholders_t, _impl::l_get_arg_storage_pair_type >::type
            arg_storage_pair_list_t;

        // Wrap the meta datas in pointer-to-const types
        typedef typename boost::mpl::transform< metadata_vector_t,
            pointer< boost::add_const< boost::mpl::_1 > > >::type::type metadata_ptr_vector_t;

        typedef _impl::compute_index_set< placeholders_t > check_holes;
        typedef typename check_holes::raw_index_list index_list;
        typedef typename check_holes::index_set index_set;

        // actual check if the user specified placeholder arguments with the same index
        GRIDTOOLS_STATIC_ASSERT((len <= boost::mpl::size< index_set >::type::value),
            "you specified two different placeholders with the same index, which is not allowed. check the arg "
            "definitions.");
        GRIDTOOLS_STATIC_ASSERT((len >= boost::mpl::size< index_set >::type::value), "something strange is happening.");

        /**
         * Type of fusion::vector of pointers to storages as indicated in Placeholders
         */
        typedef typename boost::fusion::result_of::as_vector< arg_storage_pair_list_t >::type
            arg_storage_pair_fusion_list_t;

        /**
           Wrapper for a fusion set of pointers (built from an MPL sequence) containing the
           metadata information for the storages.
         */
        typedef metadata_set< metadata_ptr_vector_t > metadata_set_t;

        /**
         * fusion::vector of pointers to storages
         */
        arg_storage_pair_fusion_list_t m_arg_storage_pair_list;

        /**
           tuple of pointers to the storages metadata. Not that metadata is constant,
           so storing its original pointer is not needed
         */
        metadata_set_t m_metadata_set;

        template < typename... DataStores >
        aggregator_type(DataStores &... ds) : m_arg_storage_pair_list(), m_metadata_set() {

            typedef typename boost::mpl::and_< boost::mpl::or_< is_data_store< DataStores >,
                is_data_store_field< DataStores > >... >::type is_valid_element_pack;

            GRIDTOOLS_STATIC_ASSERT(is_valid_element_pack::value, "wrong type");
            GRIDTOOLS_STATIC_ASSERT((sizeof...(DataStores) > 0),
                "Computations with no data_stores are not supported. "
                "Add at least one data_store to the aggregator_type "
                "definition.");

            GRIDTOOLS_STATIC_ASSERT(
                (boost::mpl::size< placeholders_t >::value -
                        boost::mpl::count_if< placeholders_t, is_tmp_arg< boost::mpl::_ > >::value ==
                    sizeof...(DataStores)),
                "The number of arguments specified when constructing the aggregator_type is not the same as the number "
                "of "
                "args to non-temporaries. Double check the temporary flag in the arg types or add the "
                "necessary storages.");
            _impl::fill_metadata_set< metadata_set_t >(m_metadata_set).reassign(ds...);

            // create a fusion vector that contains all the arg_storage_pairs to all non temporary args
            typedef boost::mpl::filter_view< placeholders_t, boost::mpl::not_< is_tmp_arg< boost::mpl::_1 > > >
                non_tmp_placeholders;
            typedef typename boost::mpl::fold< non_tmp_placeholders,
                boost::mpl::vector0<>,
                boost::mpl::push_back< boost::mpl::_1, _impl::get_arg_storage_pair_type< boost::mpl::_2 > > >::type
                non_tmp_arg_storage_pairs_mpl_vec;
            typedef typename boost::fusion::result_of::as_vector< non_tmp_arg_storage_pairs_mpl_vec >::type
                non_tmp_arg_storage_pairs_fusion_vec;
            // initialize those arg_storage_pairs with the given data_stores
            non_tmp_arg_storage_pairs_fusion_vec tmp_arg_storage_pair_vec(&ds...);

            // create a filter view to filter all the non temporary arg_storage_pairs from m_arg_storage_pair_list
            // and initialize with the previously created temporary vector.
            boost::fusion::filter_view< arg_storage_pair_fusion_list_t,
                boost::mpl::not_< is_arg_storage_pair_to_tmp< boost::mpl::_ > > >
                filtered_vals(m_arg_storage_pair_list);
            boost::fusion::copy(tmp_arg_storage_pair_vec, filtered_vals);
        }

        aggregator_type(aggregator_type const &other)
            : m_arg_storage_pair_list(other.m_arg_storage_pair_list), m_metadata_set(other.m_metadata_set) {}

        void print() {
            printf("aggregator_type: Storage pointers\n");
            boost::fusion::for_each(m_arg_storage_pair_list, _debug::print_type());
            printf("aggregator_type: Metadata set pointers\n");
            boost::fusion::for_each(m_metadata_set.sequence_view(), _debug::print_type());
            printf("aggregator_type: End info\n");
        }

        /**
         * @brief returning by const reference the metadata set
         */
        metadata_set_t const &get_metadata_set() const { return m_metadata_set; }

        /**
         * @brief returning by const reference the arg_storage_pairs
         */
        arg_storage_pair_fusion_list_t const &get_arg_storage_pairs() const { return m_arg_storage_pair_list; }

        /**
         *  @brief given the placeholder type returns the corresponding arg_storage_pair by const reference
         */
        template < typename StoragePlaceholder, 
            typename RealStoragePlaceholder = 
                typename boost::mpl::if_< 
                    is_tmp_arg<StoragePlaceholder>, 
                    typename _impl::replace_arg_storage_info< tmp_storage_info_id_t, StoragePlaceholder >::type, 
                    StoragePlaceholder 
                >::type >
        typename _impl::get_arg_storage_pair_type< RealStoragePlaceholder >::type const &get_arg_storage_pair() const {
            return boost::fusion::deref(
                boost::fusion::find< typename _impl::get_arg_storage_pair_type< RealStoragePlaceholder >::type >(
                    m_arg_storage_pair_list));
        }

        /**
         *  @brief given the placeholder type and a data_store pointer, this method initializes the corresponding
         * arg_storage_pair
         */
        template < typename StoragePlaceholder >
        void set_arg_storage_pair(pointer< typename StoragePlaceholder::storage_t > storagePtr) {
            typename _impl::get_arg_storage_pair_type< StoragePlaceholder >::type tmp(storagePtr.get());
            boost::fusion::deref(
                boost::fusion::find< typename _impl::get_arg_storage_pair_type< StoragePlaceholder >::type >(
                    m_arg_storage_pair_list)) = tmp;
            _impl::fill_metadata_set< metadata_set_t >(m_metadata_set).reassign(storagePtr);
        }

        template < typename... DataStores >
        void reassign(DataStores &... stores) {
            typedef typename boost::mpl::and_< boost::mpl::or_< is_data_store< DataStores >,
                is_data_store_field< DataStores > >... >::type is_valid_element_pack;

            GRIDTOOLS_STATIC_ASSERT(is_valid_element_pack::value, "wrong type");
            GRIDTOOLS_STATIC_ASSERT((sizeof...(DataStores) > 0),
                "the assign_pointers must be called with at least one argument. "
                "otherwise what are you calling it for?");
            _impl::fill_metadata_set< metadata_set_t >(m_metadata_set).reassign(stores...);

            // create a fusion vector that contains all the arg_storage_pairs to all non temporary args
            typedef boost::mpl::filter_view< placeholders_t, boost::mpl::not_< is_tmp_arg< boost::mpl::_1 > > >
                non_tmp_placeholders;
            typedef typename boost::mpl::fold< non_tmp_placeholders,
                boost::mpl::vector0<>,
                boost::mpl::push_back< boost::mpl::_1, _impl::get_arg_storage_pair_type< boost::mpl::_2 > > >::type
                non_tmp_arg_storage_pairs_mpl_vec;
            typedef typename boost::fusion::result_of::as_vector< non_tmp_arg_storage_pairs_mpl_vec >::type
                non_tmp_arg_storage_pairs_fusion_vec;
            // initialize those arg_storage_pairs with the given data_stores
            non_tmp_arg_storage_pairs_fusion_vec tmp_arg_storage_pair_vec(&stores...);

            // create a filter view to filter all the non temporary arg_storage_pairs from m_arg_storage_pair_list
            // and initialize with the previously created temporary vector.
            boost::fusion::filter_view< arg_storage_pair_fusion_list_t,
                boost::mpl::not_< is_arg_storage_pair_to_tmp< boost::mpl::_ > > >
                filtered_vals(m_arg_storage_pair_list);
            boost::fusion::copy(tmp_arg_storage_pair_vec, filtered_vals);
        }
    };

    template < typename domain >
    struct is_aggregator_type : boost::mpl::false_ {};

    template < typename Placeholders >
    struct is_aggregator_type< aggregator_type< Placeholders > > : boost::mpl::true_ {};

    template < uint_t... Indices, typename... Storages >
    aggregator_type< boost::mpl::vector< arg< Indices, Storages >... > > instantiate_aggregator_type(
        gt_integer_sequence< uint_t, Indices... > seq_, Storages &... storages_) {
        auto dom_ = aggregator_type< boost::mpl::vector< arg< Indices, Storages >... > >(
            boost::fusion::make_vector(&storages_...));
        return dom_;
    }

    template < typename... Storage >
    auto make_aggregator_type(Storage &... storages_) -> decltype(
        instantiate_aggregator_type(make_gt_integer_sequence< uint_t, sizeof...(Storage) >(), storages_...)) {
        return instantiate_aggregator_type(make_gt_integer_sequence< uint_t, sizeof...(Storage) >(), storages_...);
    }

} // namespace gridtools

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

#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/container/vector/convert.hpp>
#include <boost/fusion/include/as_set.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/copy.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/push_back.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/fusion/include/join.hpp>
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

#include "../common/generic_metafunctions/gt_integer_sequence.hpp"
#include "../common/generic_metafunctions/is_sequence_of.hpp"
#include "../common/generic_metafunctions/is_variadic_pack_of.hpp"
#include "../common/generic_metafunctions/static_if.hpp"
#include "../common/generic_metafunctions/variadic_to_vector.hpp"

#include "../common/make_from_permutation.hpp"
#include "../common/metadata_set.hpp"
#include "../storage/storage-facility.hpp"

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
        typedef Placeholders placeholders_t;

        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< placeholders_t, is_arg >::type::value),
            "wrong type: the aggregator_type template argument must be an MPL vector of placeholders (arg<...>)");

        const static uint_t len = boost::mpl::size< placeholders_t >::type::value;

        // filter out the storage infos which are the same
        typedef typename boost::mpl::fold< placeholders_t,
            boost::mpl::set0<>,
            boost::mpl::insert< boost::mpl::_1, get_storage_info_from_arg< boost::mpl::_2 > > >::type
            mpl_metadata_set_t;

        // create an mpl::vector of metadata types
        typedef typename boost::mpl::copy< mpl_metadata_set_t,
            boost::mpl::back_inserter< boost::mpl::vector0<> > >::type metadata_vector_t;
        const static uint_t len_meta = boost::mpl::size< metadata_vector_t >::type::value;

        // Get a sequence of the same type of placeholders_t, but containing the storage types for each placeholder
        typedef typename boost::mpl::transform< placeholders_t, _impl::l_get_arg_storage_pair_type >::type
            arg_storage_pair_list_t;

        // Wrap the meta datas in pointer-to-const types
        typedef typename boost::mpl::transform< metadata_vector_t,
            pointer< boost::add_const< boost::mpl::_1 > > >::type::type metadata_ptr_vector_t;

        /**
         * Type of fusion::vector of pointers to storages as indicated in Placeholders
         */
        typedef
            typename boost::fusion::result_of::as_set< arg_storage_pair_list_t >::type arg_storage_pair_fusion_list_t;

        using tmp_arg_storage_pair_fusion_list_t = typename boost::fusion::result_of::as_set<
            typename boost::mpl::filter_view< arg_storage_pair_list_t, is_tmp_arg< boost::mpl::_ > >::type >::type;

        using non_tmp_arg_storage_pair_fusion_list_t =
            typename boost::fusion::result_of::as_set< typename boost::mpl::filter_view< arg_storage_pair_list_t,
                boost::mpl::not_< is_tmp_arg< boost::mpl::_ > > >::type >::type;
        /**
           Wrapper for a fusion set of pointers (built from an MPL sequence) containing the
           metadata information for the storages.
         */
        typedef metadata_set< metadata_ptr_vector_t > metadata_set_t;

        /**
         * fusion::vector of pointers to storages
         */
        arg_storage_pair_fusion_list_t m_arg_storage_pair_list =
            _impl::default_host_container< arg_storage_pair_fusion_list_t >();

        /**
           tuple of pointers to the storages metadata. Note that metadata is constant,
           so storing its original pointer is not needed
         */
        metadata_set_t m_metadata_set;

        /**
           This constructor takes a variadic list of arg_storage_pairs and assigns the contents
           to the arg_storage_pair_fusion_list_t element. Also the metadata_set is updated properly.
         */
        template < typename... ArgStoragePairs,
            typename boost::enable_if< typename _impl::aggregator_arg_storage_pair_check<
                                           typename std::decay< ArgStoragePairs >::type... >::type,
                int >::type = 0 >
        aggregator_type(ArgStoragePairs &&... arg_storage_pairs)
            : aggregator_type(_impl::private_ctor,
                  boost::fusion::make_vector(std::forward< ArgStoragePairs >(arg_storage_pairs)...)) {}

        /**
           This constructor takes a variadic list of data_stores, data_store_fields, or std::vectors and assigns
           arg_storage_pairs
           filled with the correct pointers to the arg_storage_pair_fusion_list_t element. Also the metadata_set is
           updated properly.
         */
        template < typename... DataStores,
            typename boost::enable_if<
                typename _impl::aggregator_storage_check< typename std::decay< DataStores >::type... >::type,
                int >::type = 0 >
        aggregator_type(DataStores &&... ds)
            : aggregator_type(
                  _impl::private_ctor, non_tmp_arg_storage_pair_fusion_list_t{std::forward< DataStores >(ds)...}) {}

        aggregator_type(aggregator_type const &other) : aggregator_type(other.m_arg_storage_pair_list) {}
        aggregator_type(aggregator_type &&other) noexcept : aggregator_type(std::move(other.m_arg_storage_pair_list)) {}

        aggregator_type &operator=(aggregator_type const &other) {
            m_arg_storage_pair_list = other.m_arg_storage_pair_list;
            update_metadata_set();
            return *this;
        }
        aggregator_type &operator=(aggregator_type &&other) noexcept {
            m_arg_storage_pair_list = std::move(other.m_arg_storage_pair_list);
            update_metadata_set();
            return *this;
        }

        void print() const {
            printf("aggregator_type: Storage pointers\n");
            boost::fusion::for_each(m_arg_storage_pair_list, _debug::print_type());
            printf("aggregator_type: Metadata set pointers\n");
            boost::fusion::for_each(m_metadata_set.sequence_view(), _debug::print_type());
            printf("aggregator_type: End info\n");
        }

        /**
         * @brief returning by const reference the set of all metadatas
         */
        metadata_set_t const &metadata_set_view() const { return m_metadata_set; }

        /**
         * @brief returning by reference the list of all arg storage pairs. An arg storage pair maps
         * an arg to an instance of a data_store, data_store_field, or std::vector.
         */
        arg_storage_pair_fusion_list_t &get_arg_storage_pairs() { return m_arg_storage_pair_list; }
        const arg_storage_pair_fusion_list_t &get_arg_storage_pairs() const { return m_arg_storage_pair_list; }

        /**
         * @brief returning by const reference an arg storage pair. An arg storage pair maps
         * an arg to an instance of a data_store, data_store_field, or std::vector.
         */
        template < typename StoragePlaceholder >
        typename _impl::create_arg_storage_pair_type< StoragePlaceholder >::type const &get_arg_storage_pair() const {
            return at< typename _impl::create_arg_storage_pair_type< StoragePlaceholder >::type >();
        }

        /**
         *  @brief given the placeholder type and a data_store pointer, this method initializes the corresponding
         * arg_storage_pair that maps the arg to an instance of either a data_store, data_store_field, or std::vector.
         */
        template < typename StoragePlaceholder >
        void set_arg_storage_pair(typename StoragePlaceholder::storage_t &&storage) {
            at< typename _impl::create_arg_storage_pair_type< StoragePlaceholder >::type >().m_value =
                std::move(storage);
            update_metadata_set();
        }

        template < typename... DataStores,
            typename boost::enable_if<
                typename _impl::aggregator_storage_check< typename std::decay< DataStores >::type... >::type,
                int >::type = 0 >
        void reassign_storages_impl(DataStores &&... stores) {
            GRIDTOOLS_STATIC_ASSERT(
                (sizeof...(DataStores) == boost::mpl::size< non_tmp_arg_storage_pair_fusion_list_t >{}),
                "the reassign_storages_impl must be called with at least one argument. "
                "otherwise what are you calling it for?");
            // create a fusion vector that contains all the arg_storage_pairs to all non temporary args
            // initialize those arg_storage_pairs with the given data_stores
            non_tmp_arg_storage_pair_fusion_list_t tmp_arg_storage_pair_vec(std::forward< DataStores >(stores)...);

            // create a filter view to filter all the non temporary arg_storage_pairs from m_arg_storage_pair_list
            // and initialize with the previously created temporary vector.
            boost::fusion::filter_view< arg_storage_pair_fusion_list_t,
                boost::mpl::not_< is_tmp_arg< boost::mpl::_ > > > filtered_vals(m_arg_storage_pair_list);
            boost::fusion::copy(tmp_arg_storage_pair_vec, filtered_vals);
            update_metadata_set();
        }

        template < typename... ArgStoragePairs,
            typename boost::enable_if< typename _impl::aggregator_arg_storage_pair_check< ArgStoragePairs... >::type,
                int >::type = 0 >
        void reassign_arg_storage_pairs_impl(ArgStoragePairs &&... arg_storage_pairs) {
            _impl::fill_arg_storage_pair_list< arg_storage_pair_fusion_list_t >{m_arg_storage_pair_list}.reassign(
                std::forward< ArgStoragePairs >(arg_storage_pairs)...);
            update_metadata_set();
        }

      private:
        explicit aggregator_type(arg_storage_pair_fusion_list_t const &src) : m_arg_storage_pair_list(src) {
            update_metadata_set();
        }
        explicit aggregator_type(arg_storage_pair_fusion_list_t &&src) noexcept
            : m_arg_storage_pair_list(std::move(src)) {
            update_metadata_set();
        }

        template < typename NonTmp, typename Tmp = tmp_arg_storage_pair_fusion_list_t >
        aggregator_type(_impl::private_ctor_t, NonTmp &&non_tmp, Tmp &&tmp = _impl::default_host_container< Tmp >())
            : aggregator_type(make_from_permutation< arg_storage_pair_fusion_list_t >(
                  _impl::make_joint_view(std::forward< NonTmp >(non_tmp), std::forward< Tmp >(tmp)))) {}

        void update_metadata_set() {
            m_metadata_set = {};
            boost::fusion::for_each(
                m_arg_storage_pair_list, _impl::add_to_metadata_set< metadata_set_t >{m_metadata_set});
        }

        template < typename Key >
        Key &at() {
            return boost::fusion::at_key< Key >(m_arg_storage_pair_list);
        }
        template < typename Key >
        Key const &at() const {
            return boost::fusion::at_key< Key >(m_arg_storage_pair_list);
        }
    };

    template < typename domain >
    struct is_aggregator_type : boost::mpl::false_ {};

    template < typename Placeholders >
    struct is_aggregator_type< aggregator_type< Placeholders > > : boost::mpl::true_ {};

} // namespace gridtools

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
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/copy.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/push_back.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/fusion/mpl.hpp>
#include <boost/fusion/view/filter_view.hpp>
#include <boost/mpl/find.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/sort.hpp>
#include <boost/mpl/transform.hpp>
#include <iosfwd>

#include "../common/generic_metafunctions/arg_comparator.hpp"
#include "../common/generic_metafunctions/is_variadic_pack_of.hpp"
#include "../common/generic_metafunctions/static_if.hpp"
#include "../common/generic_metafunctions/variadic_to_vector.hpp"
#include "../common/gpu_clone.hpp"

#include "../storage/metadata_set.hpp"
#include "../storage/storage.hpp"
#include "../storage/storage_functors.hpp"

#include "aggregator_type_impl.hpp"
#include "arg.hpp"
#include "arg_metafunctions.hpp"

/**@file
   @brief This file contains the global list of placeholders to the storages
*/
namespace gridtools {

    namespace _impl {
        // metafunction to extract the storage type from the pointer
        template < typename T, typename U >
        struct matches {
            typedef typename boost::is_same< typename T::value_type, U >::type type;
        };
    }

    // fwd declaration
    template < typename T >
    struct is_arg;

    /**
       @brief This struct contains the global list of placeholders to the storages
     * @tparam Placeholders list of placeholders of type arg<I,T>

     NOTE: Note on the terminology: we call "global" the quantities having the "computation" granularity,
     and "local" the quantities having the "ESF" or "MSS" granularity. This class holds the global list
     of placeholders, i.e. all the placeholders used in the current computation. This list will be
     split into several local lists, one per ESF (or fused MSS).

     This class reorders the placeholders according to their indices, checks that there's no holes in the numeration,
    */

    template < typename Placeholders >
    struct aggregator_type : public clonable_to_gpu< aggregator_type< Placeholders > > {

        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< Placeholders >::type::value > 0),
            "The aggregator_type must be constructed with at least one storage placeholder. If you don't use any "
            "storage "
            "you are probably trying to do something which is not a stencil operation, aren't you?");
        typedef typename boost::mpl::sort< Placeholders, arg_comparator >::type placeholders_t;

        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< placeholders_t, is_arg >::type::value), "wrong type:\
 the aggregator_type template argument must be an MPL vector of placeholders (arg<...>)");

      private:
        BOOST_STATIC_CONSTANT(uint_t, len = boost::mpl::size< placeholders_t >::type::value);

        // filter out the metadatas which are the same
        typedef typename boost::mpl::fold< placeholders_t,
            boost::mpl::set0<> // check if the argument is a storage placeholder before extracting the metadata
            ,
            boost::mpl::if_< is_storage_arg< boost::mpl::_2 >,
                                               boost::mpl::insert< boost::mpl::_1, arg2metadata< boost::mpl::_2 > >,
                                               boost::mpl::_1 > >::type original_metadata_set_t;

        /** @brief Create an mpl::vector of metadata types*/
        typedef typename boost::mpl::fold< original_metadata_set_t,
            boost::mpl::vector0<>,
            boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 > >::type original_metadata_t;

        BOOST_STATIC_CONSTANT(uint_t, len_meta = boost::mpl::size< original_metadata_t >::type::value);

        /**
         * \brief Get a sequence of the same type of placeholders_t, but containing the storage types for each
         * placeholder
         * \todo I would call it instead of l_get_type l_get_storage_type
         */
        typedef typename boost::mpl::transform< placeholders_t, _impl::l_get_type >::type storage_list;

        /**
         * \brief Get a sequence of the same type of placeholders_t, but containing the iterator types corresponding to
         * the placeholder's storage type
         */
        typedef typename boost::mpl::transform< placeholders_t, _impl::l_get_it_type >::type iterators_list_mpl;

        /** @brief Wrap the meta datas in pointer-to-const types*/
        typedef typename boost::mpl::transform< original_metadata_t,
            pointer< boost::add_const< boost::mpl::_1 > > >::type::type mpl_metadata_ptr_list;

      public:
        typedef _impl::compute_index_set< placeholders_t > check_holes;
        typedef typename check_holes::raw_index_list index_list;
        typedef typename check_holes::index_set index_set;

        // actual check if the user specified placeholder arguments with the same index
        GRIDTOOLS_STATIC_ASSERT((len <= boost::mpl::size< index_set >::type::value),
            "you specified two different placeholders with the same index, which is not allowed. check the arg "
            "defiintions.");
        GRIDTOOLS_STATIC_ASSERT((len >= boost::mpl::size< index_set >::type::value), "something strange is happening.");

        /**
           @brief MPL vector of storage pointers
         */
        typedef storage_list arg_list_mpl;

        /**
           @brief MPL vector of placeholders

           template argument in the class definition reordered according to the arg index
         */
        typedef placeholders_t placeholders;

        /**
         * Type of fusion::vector of pointers to storages as indicated in Placeholders
         */
        typedef typename boost::fusion::result_of::as_vector< arg_list_mpl >::type arg_list;

        /**
         * Type of fusion::set of pointers to meta storages
         */
        typedef typename boost::fusion::result_of::as_set< mpl_metadata_ptr_list >::type metadata_ptr_list;
        /**
         * Type of fusion::vector of pointers to iterators as indicated in Placeholders
         */
        typedef typename boost::fusion::result_of::as_vector< iterators_list_mpl >::type iterator_list;

        /**
           Wrapper for a fusion set of pointers (built from an MPL sequence) containing the
           metadata information for the storages.
         */
        typedef metadata_set< metadata_ptr_list > metadata_set_t;

        /**
         * fusion::vector of pointers to storages
         */
        arg_list m_storage_pointers;

        /**
         * fusion::vector of pointers to storages before the updates needed before the start of the computation
         */
        arg_list m_original_pointers;

        /**
           tuple of pointers to the storages metadata. Not that metadata is constant,
           so storing its original pointer is not needed

         */
        metadata_set_t m_metadata_set;

#ifdef CXX11_ENABLED

        template < typename MetaDataSequence >
        void assign_pointers(MetaDataSequence &) {}

        /**@brief recursively assignes all the pointers passed as arguments to storages.
         */
        template < typename MetaDataSequence, typename ArgStoragePair0, typename... OtherArgs >
        typename boost::enable_if_c< is_any_storage< typename ArgStoragePair0::storage_type >::type::value, void >::type
        assign_pointers(MetaDataSequence &sequence_, ArgStoragePair0 arg0, OtherArgs... other_args) {
            assert(arg0.ptr.get());
            boost::fusion::at< typename ArgStoragePair0::arg_type::index_t >(m_storage_pointers) = arg0.ptr;
            // storing the value of the pointers in a 'backup' fusion vector
            boost::fusion::at< typename ArgStoragePair0::arg_type::index_t >(m_original_pointers) = arg0.ptr;
            if (!sequence_
                     .template present< pointer< const typename ArgStoragePair0::storage_type::storage_info_type > >())
                sequence_.insert(pointer< const typename ArgStoragePair0::storage_type::storage_info_type >(
                    &(arg0.ptr->meta_data())));
            assign_pointers(sequence_, other_args...);
        }

        /** overload for non-storage types (i.e. arbitrary user-defined types which
            do not contain necessarily a meta_data)*/
        template < typename MetaDataSequence, typename ArgStoragePair0, typename... OtherArgs >
        typename boost::disable_if_c< is_any_storage< typename ArgStoragePair0::storage_type >::type::value,
            void >::type
        assign_pointers(MetaDataSequence &sequence_, ArgStoragePair0 arg0, OtherArgs... other_args) {
            assert(arg0.ptr.get());
            boost::fusion::at< typename ArgStoragePair0::arg_type::index_t >(m_storage_pointers) = arg0.ptr;
            boost::fusion::at< typename ArgStoragePair0::arg_type::index_t >(m_original_pointers) = arg0.ptr;
            assign_pointers(sequence_, other_args...);
        }

#endif
      public:
#if defined(CXX11_ENABLED)
        /** @brief variadic constructor
            construct the aggregator_type given an arbitrary number of placeholders to the non-temporary
            storages passed as arguments.

            USAGE EXAMPLE:
            \verbatim
            aggregator_type((p1=storage_1), (p2=storage_2), (p3=storage_3));
            \endverbatim
        */
        template < typename... Pairs >
        aggregator_type(Pairs... pairs_) : m_storage_pointers(), m_metadata_set() {

            GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(is_arg_storage_pair< Pairs >::value...), "wrong type");
            GRIDTOOLS_STATIC_ASSERT((sizeof...(Pairs) > 0),
                "Computations with no storages are not supported. "
                "Add at least one storage to the aggregator_type "
                "definition.");

            typedef boost::fusion::filter_view< arg_list, is_not_tmp_storage< boost::mpl::_1 > > view_type;

            GRIDTOOLS_STATIC_ASSERT((boost::fusion::result_of::size< view_type >::type::value == sizeof...(Pairs)),
                "The number of arguments specified when constructing the domain_type is not the same as the number of "
                "placeholders "
                "to non-temporary storages. Double check the temporary flag in the meta_storage types or add the "
                "necessary storages.");

            // So far we checked that the number of arguments provided
            // match with the expected number of non-temporaries and
            // that there is at least one argument (no-default
            // constructor syntax). Now we need to check that all the
            // placeholders used in the processes are valid. To do so
            // we use a set. We insert arguments into a set so that we
            // can identify if a certain argument type appears
            // twice. (In the arg_storage_pair we check that the
            // storage types are the same between the arg and the
            // storage). This should be sufficient to prove that the
            // argument list is valid. It is in principle possible
            // that someone passes a placeholder to a temporary and
            // associates it to a user-instantiated temporary pointer,
            // but this is very complicated and I don't think we
            // should check for this.
            typedef typename variadic_to_vector< typename Pairs::arg_type... >::type v_args;
            typedef typename boost::mpl::fold< v_args,
                boost::mpl::set0<>,
                boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 > >::type counting_map;

            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< counting_map >::type::value == sizeof...(Pairs)),
                "Some placeholders appear to be used more than once in the association between placeholders and "
                "storages");

            assign_pointers(m_metadata_set, pairs_...);
        }
#endif

        /**empty functor*/
        struct empty {
            void operator()() const {}
        };

        /**
           @brief functor to insert a boost fusion sequence to the metadata set
           @tparam Sequence is of type metadata_set

           to be used whithin boost::mpl::for_each
         */
        template < typename Sequence >
        struct assign_metadata_set {
            GRIDTOOLS_STATIC_ASSERT(is_metadata_set< Sequence >::value, "Internal error: wrong type");

          private:
            Sequence &m_sequence;

          public:
            assign_metadata_set(Sequence &sequence_) : m_sequence(sequence_) {}

            /** @brief operator inserting a storage raw pointer

                filters out the arguments which are not of storage type (and thus do not have an associated metadata)
             */
            template < typename Arg >
            void operator()(Arg const *arg_) const {
                // filter out the arguments which are not of storage type (and thus do not have an associated metadata)
                static_if< is_actual_storage< pointer< Arg > >::type::value >::eval(
                    insert_if_not_present< Sequence, Arg >(m_sequence, *arg_), empty());
            }

            /** @brief operator registering the storage info object, given a raw pointer

                specialization for the case in which the storage is a std::vector of storages sharing
                the same storage info
             */
            template < typename Arg >
            void operator()(std::vector< pointer< Arg > > const *arg_) const {
                // filter out the arguments which are not of storage type (and thus do not have an associated metadata)
                static_if< is_actual_storage< pointer< Arg > >::type::value >::eval(
                    insert_if_not_present< Sequence, Arg >(m_sequence, *(*arg_)[0]), empty());
            }

#ifdef CXX11_ENABLED

            /** @brief operator geristering the storage info object given a storage gridtools::pointer

                filters out the arguments which are not of storage type (and thus do not have an associated metadata)
             */
            template < typename Arg >
            void operator()(pointer< Arg > const &arg_) const {
                // filter out the arguments which are not of storage type (and thus do not have an associated metadata)
                if (arg_.get()) // otherwise it's no_storage_type_yet
                    static_if< is_actual_storage< pointer< Arg > >::type::value >::eval(
                        insert_if_not_present< Sequence, Arg >(m_sequence, *arg_), empty());
            }

            /** @brief operator registering the storage info object, given a raw pointer

                specialization for the case in which the storage is a std::vector of storages sharing
                the same storage info
            */
            template < typename Arg >
            void operator()(pointer< std::vector< pointer< Arg > > > const &arg_) const {
                // filter out the arguments which are not of storage type (and thus do not have an associated metadata)
                if (arg_.get()) // otherwise it's no_storage_type_yet
                    static_if< is_actual_storage< pointer< Arg > >::type::value >::eval(
                        insert_if_not_present< Sequence, Arg >(m_sequence, (*arg_)[0]), empty());
            }

            template < typename Arg, uint_t Size >
            void operator()(pointer< expandable_parameters< Arg, Size > > const &arg_) const {
                // filter out the arguments which are not of storage type (and thus do not have an associated metadata)
                if (arg_.get()) // otherwise it's no_storage_type_yet
                    static_if< is_actual_storage< pointer< Arg > >::type::value >::eval(
                        insert_if_not_present< Sequence, expandable_parameters< Arg, Size > >(m_sequence, *arg_),
                        empty());
            }
#endif
        };

/**@brief Constructor from boost::fusion::vector of raw pointers
 * @tparam RealStorage fusion::vector of raw pointers to storages sorted with increasing indices of the placeholders
 * @param real_storage The actual fusion::vector with the values
 TODO: when I have only one placeholder and C++11 enabled this constructor is erroneously picked
 */
#ifdef CXX11_ENALBED
        template < template < typename... > class Vector, typename... Storages >
        explicit aggregator_type(Vector< Storages *... > const &real_storage_)
#else
        template < typename RealStorage >
        explicit aggregator_type(RealStorage const &real_storage_)
#endif
            : m_storage_pointers(), m_metadata_set() {

            // TODO: how to check the assertion below?
            // #ifdef CXX11_ENABLED
            //             GRIDTOOLS_STATIC_ASSERT(is_fusion_vector<RealStorage>::value, "the argument passed to the
            //             domain type constructor must be a fusion vector, or a pair (placeholder = storage), see the
            //             aggregator_type constructors");
            // #endif

            typedef boost::fusion::filter_view< arg_list, is_not_tmp_storage< boost::mpl::_1 > > view_type;

            view_type fview(m_storage_pointers);
            GRIDTOOLS_STATIC_ASSERT(boost::fusion::result_of::size< view_type >::type::value ==
                                        boost::mpl::size< RealStorage >::type::value,
                "The number of arguments specified when constructing the aggregator_type is not the same as the number "
                "of "
                "placeholders "
                "to non-temporary storages. Double check the temporary flag in the meta_storage types or add the "
                "necessary storages.");

            // below few metafunctions only to protect the user from mismatched storages
            typedef typename boost::mpl::fold< arg_list_mpl,
                boost::mpl::vector0<>,
                boost::mpl::if_< is_not_tmp_storage< boost::mpl::_2 >,
                                                   boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 >,
                                                   boost::mpl::_1 > >::type view_type_mpl;

            // checking thet all the storages are correct
            typedef typename boost::mpl::fold<
                boost::mpl::range_c< uint_t, 0, boost::mpl::size< RealStorage >::type::value >,
                boost::mpl::bool_< true >,
                boost::mpl::and_< _impl::matches< boost::mpl::at< view_type_mpl, boost::mpl::_2 >,
                                      boost::remove_pointer< boost::mpl::at< RealStorage, boost::mpl::_2 > > >,
                    boost::mpl::_1 > >::type::type storages_matching;

            GRIDTOOLS_STATIC_ASSERT(storages_matching::value,
                "Error in the definition of the aggregator_type. The storage type associated to one of the \'arg\' "
                "types "
                "is not the correct one. Check that the storage_type used when defining each \'arg\' matches the "
                "corresponding storage passed as run-time argument of the aggregator_type constructor");

            // NOTE: an error in the line below could mean that the storage type
            // associated to the arg is not the
            // correct one (TODO: think of a good way to catch this)
            // copy of the non-tmp storages into m_storage_pointers
            boost::fusion::copy(real_storage_, fview);

            // copy of the non-tmp metadata into m_metadata_set
            boost::fusion::for_each(real_storage_, assign_metadata_set< metadata_set_t >(m_metadata_set));

#ifdef VERBOSE
            std::cout << "\nThese are the view values" << boost::fusion::size(fview) << std::endl;
            boost::fusion::for_each(m_storage_pointers, _debug::dt_print_pointer());
#endif
            view_type original_fview(m_original_pointers);
            boost::fusion::copy(real_storage_, original_fview);
        }

// constructor used from whithin expandable parameters
#ifdef CXX11_ENABLED // because of std::enable_if

        /**@brief Constructor from boost::fusion::vector of gridools::pointer
         * @tparam RealStorage fusion::vector of gridtools::pointers to storages
         * @param real_storage The actual fusion::vector with the values
         TODO: when I have only one placeholder and C++11 enabled this constructor is erroneously picked
         */
        template < template < typename... > class Vector, typename... Storages >
        explicit aggregator_type(Vector< pointer< Storages >... > const &storage_pointers_)
            : m_storage_pointers(storage_pointers_), m_metadata_set() {

            boost::fusion::copy(storage_pointers_, m_original_pointers);

            // copy of the metadata into m_metadata_set
            boost::fusion::for_each(storage_pointers_, assign_metadata_set< metadata_set_t >(m_metadata_set));
        }
#endif

        /** Copy constructor to be used when cloning to GPU
         *
         * @param The object to copy. Typically this will be *this
         */
        GT_FUNCTION
        aggregator_type(aggregator_type const &other)
            : m_storage_pointers(other.m_storage_pointers), m_original_pointers(other.m_original_pointers),
              m_metadata_set(other.m_metadata_set) {}

        GT_FUNCTION
        void info() {
            printf("aggregator_type: Storage pointers\n");
            boost::fusion::for_each(m_storage_pointers, _debug::print_domain_info());
            printf("aggregator_type: Original pointers\n");
            boost::fusion::for_each(m_original_pointers, _debug::print_domain_info());
            printf("aggregator_type: End info\n");
        }

        template < typename Index >
        void storage_info(std::ostream &out_s) const {
            out_s << Index::value << " -|-> " << (boost::fusion::at_c< Index >(m_metadata_set))->name() << " "
                  << (boost::fusion::at_c< Index >(m_metadata_set))->template dim< 0 >() << "x"
                  << (boost::fusion::at_c< Index >(m_metadata_set))->template dim< 1 >() << "x"
                  << (boost::fusion::at_c< Index >(m_metadata_set))->template dim< 2 >() << ", "
                  << (boost::fusion::at_c< Index >(m_metadata_set))->strides(0) << "x"
                  << (boost::fusion::at_c< Index >(m_metadata_set))->strides(1) << "x"
                  << (boost::fusion::at_c< Index >(m_metadata_set))->strides(2) << ", \n";
        }

        /** @brief copy the pointers from the device to the host
            NOTE: no need to copy back the metadata since it has not been modified
        */
        void finalize_computation() {
            boost::fusion::for_each(m_original_pointers, call_d2h());
            boost::mpl::for_each< boost::mpl::range_c< int, 0, boost::mpl::size< arg_list >::value > >(
                copy_pointers_functor< arg_list, arg_list >(m_original_pointers, m_storage_pointers));
        }

        /**
           @brief returning by non-const reference the metadata set
        */
        metadata_set_t &metadata_set_view() { return m_metadata_set; }

        /**
           @brief returning by non-const reference the storage pointers
        */
        arg_list &storage_pointers_view() { return m_storage_pointers; }

        /**
           @brief given the placeholder type returns the corresponding storage gtidtools::pointer by reference
         */
        template < typename StoragePlaceholder >
        typename boost::mpl::at< arg_list, typename StoragePlaceholder::index_t >::type &storage_pointer() {
            return boost::fusion::at< typename StoragePlaceholder::index_t >(m_storage_pointers);
        }

        /**
           @brief given the placeholder type returns the corresponding storage gridtools::pointer by const ref
         */
        template < typename StoragePlaceholder >
        typename boost::mpl::at< arg_list, typename StoragePlaceholder::index_t >::type const &storage_pointer() const {
            return boost::fusion::at< typename StoragePlaceholder::index_t >(m_storage_pointers);
        }

        /**
           @brief metafunction returning the storage type given the placeholder type
         */
        template < typename T >
        struct storage_type {
            typedef typename boost::mpl::at< arg_list_mpl, typename T::index_t >::type::value_type type;
        };

#ifdef CXX11_ENABLED
        template < typename... Pair >
        void reassign(Pair... pairs_) {

            GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(is_arg_storage_pair< Pair >::value...), "wrong type");
            GRIDTOOLS_STATIC_ASSERT((sizeof...(Pair) > 0),
                "the assign_pointers must be called with at least one argument."
                " otherwise what are you calling it for?");
            // NOTE: the following assertion assumes there StorageArgs has length at leas 1
            // GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(is_arg_storage_pair< StorageArgs >::value...), "wrong type");
            assign_pointers(m_metadata_set, pairs_...);
        }
#endif
    };

    template < typename domain >
    struct is_aggregator_type : boost::mpl::false_ {};

    template < typename Placeholders >
    struct is_aggregator_type< aggregator_type< Placeholders > > : boost::mpl::true_ {};

#ifdef CXX11_ENABLED

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

#endif

} // namespace gridtools

#pragma once

#include <iosfwd>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/container/vector/convert.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/push_back.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/copy.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/find.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/sort.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/fusion/view/filter_view.hpp>
#include <boost/fusion/include/for_each.hpp>

#include "../common/generic_metafunctions/static_if.hpp"
#include "../common/generic_metafunctions/is_variadic_pack_of.hpp"
#include "../common/generic_metafunctions/arg_comparator.hpp"
#include "../common/gpu_clone.hpp"

#include "../storage/storage.hpp"
#include "../storage/storage_functors.hpp"
#include "../storage/metadata_set.hpp"

#include "domain_type_impl.hpp"
#include "arg_metafunctions.hpp"
#include "arg.hpp"

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
    struct domain_type : public clonable_to_gpu< domain_type< Placeholders > > {

        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< Placeholders >::type::value > 0),
            "The domain_type must be constructed with at least one storage placeholder. If you don't use any storage "
            "you are probably trying to do something which is not a stencil operation, aren't you?");
        typedef typename boost::mpl::sort< Placeholders, arg_comparator >::type placeholders_t;

        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< placeholders_t, is_arg >::type::value), "wrong type:\
 the domain_type template argument must be an MPL vector of placeholders (arg<...>)");

      private:
        BOOST_STATIC_CONSTANT(uint_t, len = boost::mpl::size< placeholders_t >::type::value);

        // filter out the metadatas which are the same
        typedef typename boost::mpl::fold< placeholders_t,
            boost::mpl::set<> // check if the argument is a storage placeholder before extracting the metadata
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
        GRIDTOOLS_STATIC_ASSERT((len == boost::mpl::size< index_set >::type::value),
            "you specified two different placeholders with the same index, which is not allowed. check the arg "
            "defiintions.");

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
            boost::fusion::at< typename ArgStoragePair0::arg_type::index_type >(m_storage_pointers) = arg0.ptr;
            // storing the value of the pointers in a 'backup' fusion vector
            boost::fusion::at< typename ArgStoragePair0::arg_type::index_type >(m_original_pointers) = arg0.ptr;
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
            boost::fusion::at< typename ArgStoragePair0::arg_type::index_type >(m_storage_pointers) = arg0.ptr;
            boost::fusion::at< typename ArgStoragePair0::arg_type::index_type >(m_original_pointers) = arg0.ptr;
            assign_pointers(sequence_, other_args...);
        }

#endif
      public:
#if defined(CXX11_ENABLED)
        /** @brief variadic constructor
            construct the domain_type given an arbitrary number of placeholders to the non-temporary
            storages passed as arguments.

            USAGE EXAMPLE:
            \verbatim
            domain_type((p1=storage_1), (p2=storage_2), (p3=storage_3));
            \endverbatim
        */
        template < typename... StorageArgs >
        domain_type(StorageArgs... args)
            : m_storage_pointers(), m_metadata_set() {

            GRIDTOOLS_STATIC_ASSERT((sizeof...(StorageArgs) > 0),
                "Computations with no storages are not supported. "
                "Add at least one storage to the domain_type "
                "definition.");
            // NOTE: the following assertion assumes there StorageArgs has length at leas 1
            GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(is_arg_storage_pair< StorageArgs >::value...), "wrong type");
            assign_pointers(m_metadata_set, args...);
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

            template < typename Arg >
            void operator()(Arg const *arg_) const {
                // filter out the arguments which are not of storage type (and thus do not have an associated metadata)
                static_if< is_actual_storage< pointer< Arg > >::type::value >::eval(
                    insert_if_not_present< Sequence, Arg >(m_sequence, *arg_), empty());
            }
        };

        /**@brief Constructor from boost::fusion::vector
         * @tparam RealStorage fusion::vector of pointers to storages sorted with increasing indices of the placeholders
         * @param real_storage The actual fusion::vector with the values
         TODO: when I have only one placeholder and C++11 enabled this constructor is erroneously picked
         */
        template < typename RealStorage >
        explicit domain_type(RealStorage const &real_storage_)
            : m_storage_pointers(), m_metadata_set() {

            // TODO: how to check the assertion below?
            // #ifdef CXX11_ENABLED
            //             GRIDTOOLS_STATIC_ASSERT(is_fusion_vector<RealStorage>::value, "the argument passed to the
            //             domain type constructor must be a fusion vector, or a pair (placeholder = storage), see the
            //             domain_type constructors");
            // #endif

            typedef boost::fusion::filter_view< arg_list, is_not_tmp_storage< boost::mpl::_1 > > view_type;

            view_type fview(m_storage_pointers);
            GRIDTOOLS_STATIC_ASSERT(boost::fusion::result_of::size< view_type >::type::value ==
                                        boost::mpl::size< RealStorage >::type::value,
                "The number of arguments specified when constructing the domain_type is not the same as the number of "
                "placeholders "
                "to non-temporary storages. Double check the temporary flag in the meta_storage types.");

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
                "Error in the definition of the domain_type. The storage type associated to one of the \'arg\' types "
                "is not the correct one. Check that the storage_type used when defining each \'arg\' matches the "
                "corresponding storage passed as run-time argument of the domain_type constructor");

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

        /** Copy constructor to be used when cloning to GPU
         *
         * @param The object to copy. Typically this will be *this
         */
        __device__ explicit domain_type(domain_type const &other)
            : m_storage_pointers(other.m_storage_pointers), m_original_pointers(other.m_original_pointers),
              m_metadata_set(other.m_metadata_set) {}

        GT_FUNCTION
        void info() {
            printf("domain_type: Storage pointers\n");
            boost::fusion::for_each(m_storage_pointers, _debug::print_domain_info());
            printf("domain_type: Original pointers\n");
            boost::fusion::for_each(m_original_pointers, _debug::print_domain_info());
            printf("domain_type: End info\n");
        }

        template < typename Index >
        void storage_info(std::ostream & out_s) const {
            out_s << Index::value << " -|-> " << (boost::fusion::at_c< Index >(m_metadata_set))->name() << " "
                  << (boost::fusion::at_c< Index >(m_metadata_set))->template dims< 0 >() << "x"
                  << (boost::fusion::at_c< Index >(m_metadata_set))->template dims< 1 >() << "x"
                  << (boost::fusion::at_c< Index >(m_metadata_set))->template dims< 2 >() << ", "
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
    };

    template < typename domain >
    struct is_domain_type : boost::mpl::false_ {};

    template < typename Placeholders >
    struct is_domain_type< domain_type< Placeholders > > : boost::mpl::true_ {};

} // namespace gridtools

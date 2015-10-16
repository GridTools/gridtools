#pragma once

#include "../common/defs.hpp"
#include "../common/gt_assert.hpp"

#include <stdio.h>
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
#include <boost/mpl/range_c.hpp>
#include <boost/fusion/view/filter_view.hpp>
#include <boost/fusion/include/for_each.hpp>
#include "gt_for_each/for_each.hpp"
#include "../common/gpu_clone.hpp"
#include "storage/storage.hpp"
#include "../storage/storage_functors.hpp"

#include "domain_type_impl.hpp"
#include "../storage/metadata_set.hpp"
#include "../common/generic_metafunctions/static_if.hpp"

/**@file
 @brief This file contains the global list of placeholders to the storages
*/
namespace gridtools {

    //fwd declaration
    template<typename T>
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

    template <typename Placeholders>
    struct domain_type : public clonable_to_gpu<domain_type<Placeholders> > {

        typedef Placeholders placeholders_t;

        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<placeholders_t, is_arg>::type::value), "wrong type:\
 the domain_type template argument must be an MPL vector of placeholders (arg<...>)");
        typedef placeholders_t original_placeholders;

    private:
        BOOST_STATIC_CONSTANT(uint_t, len = boost::mpl::size<original_placeholders>::type::value);

        // filter out the metadatas which are the same
        typedef typename boost::mpl::fold<
            original_placeholders
            , boost::mpl::set<> // check if the argument is a storage placeholder before extracting the metadata
            , boost::mpl::if_< is_storage_arg<boost::mpl::_2>
                               , boost::mpl::insert<boost::mpl::_1, arg2metadata<boost::mpl::_2> >, boost::mpl::_1 > >
        ::type original_metadata_set_t;

        /** @brief Create an mpl::vector of metadata types*/
        typedef typename boost::mpl::fold<
            original_metadata_set_t
            , boost::mpl::vector0<>
            , boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2 > >::type original_metadata_t;

        BOOST_STATIC_CONSTANT(uint_t, len_meta = boost::mpl::size<original_metadata_t>::type::value);

        /**
         * \brief Get a sequence of the same type of original_placeholders, but containing the storage types for each placeholder
         * \todo I would call it instead of l_get_type l_get_storage_type
         */
        typedef typename boost::mpl::transform<original_placeholders,
                                               _impl::l_get_type
                                               >::type raw_storage_list;


        /**
         * \brief Get a sequence of the same type of original_placeholders, but containing the iterator types corresponding to the placeholder's storage type
         */
        typedef typename boost::mpl::transform<original_placeholders,
                                               _impl::l_get_it_type
                                               >::type raw_iterators_list;

        /** @brief Wrap the meta datas in pointer-to-const types*/
        typedef typename boost::mpl::transform<original_metadata_t,
                                               pointer<boost::add_const<
                                                           boost::mpl::_1
                                                           > >
                                               >::type::type mpl_metadata_ptr_list;

    public:

        typedef _impl::compute_index_set<original_placeholders> check_holes;
        typedef typename check_holes::raw_index_list raw_index_list;
        typedef typename check_holes::index_set index_set;


        //actual check if the user specified placeholder arguments with the same index
        GRIDTOOLS_STATIC_ASSERT((len == boost::mpl::size<index_set>::type::value ), "you specified two different placeholders with the same index, which is not allowed. check the arg defiintions.");

        /**
         * \brief Definition of a random access sequence of integers between 0 and the size of the placeholder sequence
         e.g. [0,1,2,3,4]
        */
        typedef boost::mpl::range_c<uint_t ,0,len> range_t;

    private:
        typedef typename boost::mpl::find_if<raw_index_list, boost::mpl::greater<boost::mpl::_1, static_int<len-1> > >::type test;
        //check if the index list contains holes (a common error is to define a list of types with indexes which are not contiguous)
        GRIDTOOLS_STATIC_ASSERT((boost::is_same<typename test::type, boost::mpl::void_ >::value) , "the index list contains holes:\n\
The numeration of the placeholders is not contiguous. \
You have to define each arg with a unique identifier ranging from 0 to N without \"holes\".");

        /**\brief reordering vector
         * defines an mpl::vector of len indexes reordered accodring to range_t (placeholder _2 is vector<>, placeholder _1 is range_t)
         e.g.[1,3,2,4,0]
        */
        typedef typename boost::mpl::fold<range_t,
            boost::mpl::vector<>,
            boost::mpl::push_back<
                boost::mpl::_1,
                boost::mpl::find<raw_index_list, boost::mpl::_2>
            >
        >::type iter_list;

    public:

        /**\brief reordered index_list
         * Defines a mpl::vector of index::pos for the indexes in iter_list
         */
        typedef typename boost::mpl::transform<iter_list, _impl::l_get_it_pos>::type index_list;

        /**
         * \brief reordering of raw_storage_list
         creates an mpl::vector of all the storages in raw_storage_list corresponding to the indices in index_list
        */
        typedef typename boost::mpl::fold<index_list,
            boost::mpl::vector<>,
            boost::mpl::push_back<
                boost::mpl::_1,
                boost::mpl::at<raw_storage_list, boost::mpl::_2>
            >
        >::type arg_list_mpl;



        /**
         * \brief defines a reordered mpl::vector of placeholders
         */
        typedef typename boost::mpl::fold<index_list,
            boost::mpl::vector<>,
            boost::mpl::push_back<
                boost::mpl::_1,
                boost::mpl::at<original_placeholders, boost::mpl::_2>
            >
        >::type placeholders;

    private:
        typedef typename boost::mpl::fold<index_list,
            boost::mpl::vector<>,
            boost::mpl::push_back<
                boost::mpl::_1,
                boost::mpl::at<raw_iterators_list, boost::mpl::_2>
            >
        >::type iterator_list_mpl;

    public:
        /**
         * Type of fusion::vector of pointers to storages as indicated in Placeholders
         */
        typedef typename boost::fusion::result_of::as_vector<arg_list_mpl>::type arg_list;

        /**
         * Type of fusion::set of pointers to meta storages
         */
        typedef typename boost::fusion::result_of::as_set<mpl_metadata_ptr_list>::type metadata_ptr_list;
        /**
         * Type of fusion::vector of pointers to iterators as indicated in Placeholders
         */
        typedef typename boost::fusion::result_of::as_vector<iterator_list_mpl>::type iterator_list;

        /**
           Wrapper for a fusion set of pointers (built from an MPL sequence) containing the
           metadata information for the storages.
         */
        typedef metadata_set<metadata_ptr_list> metadata_set_t;

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

        template <typename MetaDataSequence>
        void assign_pointers(MetaDataSequence&) {}

        /**@brief recursively assignes all the pointers passed as arguments to storages.
         */
        template <typename MetaDataSequence, typename ArgStoragePair0, typename... OtherArgs>
        void assign_pointers(MetaDataSequence& sequence_, ArgStoragePair0 arg0, OtherArgs ... other_args)
        {
            assert(arg0.ptr);
            boost::fusion::at<typename ArgStoragePair0::arg_type::index_type>(m_storage_pointers) = arg0.ptr;
            //storing the value of the pointers in a 'backup' fusion vector
            boost::fusion::at<typename ArgStoragePair0::arg_type::index_type>(m_original_pointers) = arg0.ptr;
            if (!sequence_.template present<pointer<const typename ArgStoragePair0::storage_type::meta_data_t> >())
                sequence_.insert(pointer<const typename ArgStoragePair0::storage_type::meta_data_t>(&(arg0.ptr->meta_data())));
            assign_pointers(sequence_, other_args...);
        }

#endif
    public:

#if defined (CXX11_ENABLED)
        /** @brief variadic constructor
            construct the domain_type given an arbitrary number of placeholders to the non-temporary
            storages passed as arguments.

            USAGE EXAMPLE:
            \verbatim
            domain_type((p1=storage_1), (p2=storage_2), (p3=storage_3));
            \endverbatim
        */
        template <typename... StorageArgs>
        domain_type(StorageArgs ... args)
            : m_storage_pointers()
            , m_metadata_set()
        {
            GRIDTOOLS_STATIC_ASSERT(accumulate(logical_and(), is_arg_storage_pair<StorageArgs>::value ...), "wrong type");
            assign_pointers(m_metadata_set, args...);
        }
#endif


        /**empty functor*/
        struct empty{
            void operator()() const{
            }
        };

        /**
           @brief functor to insert a boost fusion sequence to the metadata set
           @tparam Sequence is of type metadata_set

           to be used whithin boost::mpl::for_each
         */
        template <typename Sequence>
        struct assign_metadata_set{
            GRIDTOOLS_STATIC_ASSERT(is_metadata_set<Sequence>::value, "Internal error: wrong type");
        private:
            Sequence& m_sequence;

        public:
            assign_metadata_set(Sequence& sequence_) : m_sequence(sequence_){
            }

            template <typename Arg>
            void operator()( Arg const* arg_) const{
                // filter out the arguments which are not of storage type (and thus do not have an associated metadata)
                pointer<const typename Arg::meta_data_t> ptr(&(arg_->meta_data()));
                static_if<is_storage<Arg>::type::value>::eval(
                    insert_if_not_present<Sequence, pointer<const typename Arg::meta_data_t> >
                    (m_sequence, ptr)
                    , empty());

            }
        };

        /**@brief Constructor from boost::fusion::vector
         * @tparam RealStorage fusion::vector of pointers to storages sorted with increasing indices of the pplaceholders
         * @param real_storage The actual fusion::vector with the values
         TODO: when I have only one placeholder and C++11 enabled this constructor is erroneously picked
         */
        template <typename RealStorage>
        explicit domain_type(RealStorage const & real_storage_)
            : m_storage_pointers()
            , m_metadata_set()
        {
            typedef boost::fusion::filter_view
                <arg_list,
                 is_not_tmp_storage<boost::mpl::_1> > view_type;

            view_type fview(m_storage_pointers);

            GRIDTOOLS_STATIC_ASSERT( boost::fusion::result_of::size<view_type>::type::value == boost::mpl::size<RealStorage>::type::value, "The number of arguments specified when constructing the domain_type is not the same as the number of placeholders to non-temporary storages. Double check the temporary flag in the meta_storage types.");

            typedef typename boost::mpl::fold<
                arg_list_mpl
                , boost::mpl::vector0<>
                , boost::mpl::if_<is_not_tmp_storage<boost::mpl::_2>
                                  , boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2 >
                                  , boost::mpl::_1
                                  > >
                ::type view_type_mpl;

            typedef typename boost::mpl::fold<
                boost::mpl::range_c<uint_t, 0, boost::mpl::size<RealStorage>::type::value>
                , boost::mpl::bool_<true>
                , boost::mpl::and_<
                    boost::is_same<boost::mpl::at<view_type_mpl, boost::mpl::_2>, boost::mpl::at<RealStorage, boost::mpl::_2> >
                      , boost::mpl::_1
                      >
                >::type::type storages_matching;

            GRIDTOOLS_STATIC_ASSERT(storages_matching::value, "Error in the definition of the domain_type. The storage type associated to one of the \'arg\' types is not the correct one. Check that the storage_type used when defining each \'arg\' matches the correspondent storage passed as run-time argument of the domain_type constructor");

            //NOTE: an error in the line below could mean that the storage type
            // associated to the arg is not the
            // correct one (TODO: think of a good way to catch this)
            //copy of the non-tmp storages into m_storage_pointers
            boost::fusion::copy(real_storage_, fview);

            //copy of the non-tmp metadata into m_metadata_set
            boost::fusion::for_each(real_storage_, assign_metadata_set<metadata_set_t >(m_metadata_set));

#ifdef VERBOSE
            std::cout << "\nThese are the view values" << boost::fusion::size(fview) << std::endl;
            boost::fusion::for_each(m_storage_pointers, _debug::print_pointer());
#endif
            view_type original_fview(m_original_pointers);
            boost::fusion::copy(real_storage_, original_fview);
        }

        /** Copy constructor to be used when cloning to GPU
         *
         * @param The object to copy. Typically this will be *this
         */
        __device__
        explicit domain_type(domain_type const& other)
            : m_storage_pointers(other.m_storage_pointers)
            , m_original_pointers(other.m_original_pointers)
            , m_metadata_set(other.m_metadata_set)
        { }

#ifndef NDEBUG
        GT_FUNCTION
        void info() {
            printf("domain_type: Storage pointers\n");
            boost::fusion::for_each(m_storage_pointers, _debug::print_domain_info());
            printf("domain_type: Original pointers\n");
            boost::fusion::for_each(m_original_pointers, _debug::print_domain_info());
            printf("domain_type: End info\n");
        }
#endif

        template <typename Index>
        void storage_info() const {
            std::cout << Index::value << " -|-> "
                      << (boost::fusion::at_c<Index>(m_metadata_set))->name()
                      << " "
                      << (boost::fusion::at_c<Index>(m_metadata_set))->template dims<0>()
                      << "x"
                      << (boost::fusion::at_c<Index>(m_metadata_set))->template dims<1>()
                      << "x"
                      << (boost::fusion::at_c<Index>(m_metadata_set))->template dims<2>()
                      << ", "
                      << (boost::fusion::at_c<Index>(m_metadata_set))->strides(0)
                      << "x"
                      << (boost::fusion::at_c<Index>(m_metadata_set))->strides(1)
                      << "x"
                      << (boost::fusion::at_c<Index>(m_metadata_set))->strides(2)
                      << ", "
                      << std::endl;
        }

        /** @brief copy the pointers from the device to the host
            NOTE: no need to copy back the metadata since it has not been modified
        */
        void finalize_computation() {
            boost::fusion::for_each(m_original_pointers, call_d2h());
            gridtools::for_each<
                boost::mpl::range_c<int, 0, boost::mpl::size<arg_list>::value >
                > (copy_pointers_functor<arg_list, arg_list> (m_original_pointers, m_storage_pointers));
        }

        /**
           @brief returning by non-const reference the metadata set
        */
        metadata_set_t & metadata_set_view(){return m_metadata_set;}

        /**
           @brief returning by non-const reference the storage pointers
        */
        arg_list & storage_pointers_view(){return m_storage_pointers;}

    };

    template<typename domain>
    struct is_domain_type : boost::mpl::false_ {};

    template <typename Placeholders>
    struct is_domain_type<domain_type<Placeholders> > : boost::mpl::true_{};

} // namespace gridtools

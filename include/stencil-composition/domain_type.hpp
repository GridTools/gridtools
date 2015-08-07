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
#include <storage/storage.hpp>
#include "../storage/storage_functors.hpp"

#include "domain_type_impl.hpp"

/**@file
 @brief This file contains the list of placeholders to the storages
 */

namespace gridtools {

        template<typename T>
        struct get_value_t {
            typedef typename T::value_t type;
        };

    /**
     * @tparam Placeholders list of placeholders of type arg<I,T>
     */
    template <typename Placeholders, typename MetaStorages>
    struct domain_type : public clonable_to_gpu<domain_type<Placeholders, MetaStorages> > {
        typedef Placeholders original_placeholders;
        typedef MetaStorages original_metadata;
    private:
        BOOST_STATIC_CONSTANT(uint_t, len = boost::mpl::size<original_placeholders>::type::value);
        BOOST_STATIC_CONSTANT(uint_t, len_meta = boost::mpl::size<original_metadata>::type::value);

        /**
         * \brief Get a sequence of the same type of original_placeholders, but containing the storage types for each placeholder
         * \todo I would call it instead of l_get_type l_get_storage_type
         */
        typedef typename boost::mpl::transform<original_placeholders,
                                               _impl::l_get_type
                                               >::type raw_storage_list;

        typedef typename boost::mpl::transform<original_metadata,
                                               boost::add_pointer<get_value_t<boost::mpl::_1> >
                                               >::type::type raw_metadata_list;

        /**
         * \brief Get a sequence of the same type of original_placeholders, but containing the iterator types corresponding to the placeholder's storage type
         */
        typedef typename boost::mpl::transform<original_placeholders,
                                               _impl::l_get_it_type
                                               >::type raw_iterators_list;


    public:

        typedef _impl::compute_index_set<original_placeholders> check_holes;
        typedef typename check_holes::raw_index_list raw_index_list;
        typedef typename check_holes::index_set index_set;

        typedef _impl::compute_index_set<original_metadata> check_holes_metadata;
        typedef typename check_holes_metadata::raw_index_list raw_index_list_metadata;
        typedef typename check_holes_metadata::index_set index_set_metadata;

        //actual check if the user specified placeholder arguments with the same index
        GRIDTOOLS_STATIC_ASSERT((len == boost::mpl::size<index_set>::type::value ), "you specified two different placeholders with the same index, which is not allowed. check the arg defiintions.");

        //actual check if the user specified placeholder arguments with the same index
        GRIDTOOLS_STATIC_ASSERT((len_meta == boost::mpl::size<index_set_metadata>::type::value ), "you specified two different meta_storage with the same index, which is not allowed. check the meta_storage defiintions.");

        /**
         * \brief Definition of a random access sequence of integers between 0 and the size of the placeholder sequence
         e.g. [0,1,2,3,4]
         */
        typedef boost::mpl::range_c<uint_t ,0,len> range_t;
        typedef boost::mpl::range_c<uint_t ,0,len_meta> meta_range_t;

    private:
        typedef typename boost::mpl::find_if<raw_index_list, boost::mpl::greater<boost::mpl::_1, static_int<len-1> > >::type test;
        //check if the index list contains holes (a common error is to define a list of types with indexes which are not contiguous)
        GRIDTOOLS_STATIC_ASSERT((boost::is_same<typename test::type, boost::mpl::void_ >::value) , "the index list contains holes:\n\
The numeration of the placeholders is not contiguous. You have to define each arg with a unique identifier ranging from 0 to N without \"holes\".");

        typedef typename boost::mpl::find_if<raw_index_list_metadata, boost::mpl::greater<boost::mpl::_1, static_int<len_meta-1> > >::type test2;
        //check if the index list contains holes (a common error is to define a list of types with indexes which are not contiguous)
        GRIDTOOLS_STATIC_ASSERT((boost::is_same<typename test2::type, boost::mpl::void_ >::value) , "the index list contains holes:\n\
The numeration of the meta_storages is not contiguous. You have to define each meta_storage with a unique identifier ranging from 0 to N without \"holes\".");

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

        typedef typename boost::mpl::fold<meta_range_t,
            boost::mpl::vector<>,
            boost::mpl::push_back<
                boost::mpl::_1,
                boost::mpl::find<raw_index_list_metadata, boost::mpl::_2>
            >
        >::type iter_list_metadata;

    public:

        /**\brief reordered index_list
         * Defines a mpl::vector of index::pos for the indexes in iter_list
         */
        typedef typename boost::mpl::transform<iter_list, _impl::l_get_it_pos>::type index_list;
        typedef typename boost::mpl::transform<iter_list_metadata, _impl::l_get_it_pos>::type index_list_metadata;

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

        typedef typename boost::mpl::fold<index_list_metadata,
            boost::mpl::vector<>,
            boost::mpl::push_back<
                boost::mpl::_1,
                boost::mpl::at<raw_metadata_list, boost::mpl::_2>
                >
                                          >::type::type meta_arg_list_mpl;

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

        typedef typename boost::mpl::fold<index_list_metadata,
            boost::mpl::vector<>,
            boost::mpl::push_back<
                boost::mpl::_1,
                boost::mpl::at<original_metadata, boost::mpl::_2>
            >
        >::type metadata;

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
        typedef typename boost::fusion::result_of::as_vector<meta_arg_list_mpl>::type metadata_list;
        /**
         * Type of fusion::vector of pointers to iterators as indicated in Placeholders
         */
        typedef typename boost::fusion::result_of::as_vector<iterator_list_mpl>::type iterator_list;

        /**
         * fusion::vector of pointers to storages
         */
        arg_list m_storage_pointers;

        /**
         * fusion::vector of pointers to storages before the updates needed before the start of the computation
         */
        arg_list m_original_pointers;

        /**
           tuple of pointers to the storages metadata
         */
        metadata_list m_metadata;

        /**
           tuple of pointers to the storages metadata, used to be able to delete the pointers
         */
        metadata_list m_original_metadata;

#ifdef CXX11_ENABLED
        void assign_pointers() {}

        /**@brief recursively assignes all the pointers passed as arguments to storages.
         */
        template <typename Arg0, typename... OtherArgs>
        void assign_pointers(Arg0 const& arg0, OtherArgs const& ... other_args)
        {
            boost::fusion::at<typename Arg0::arg_type::index_type>(m_storage_pointers) = arg0.ptr;
            assign_pointers(other_args...);
        }

        template <typename Arg0, typename... OtherArgs>
        void assign_metadatas(Arg0 const& arg0, OtherArgs const& ... other_args)
        {
            boost::fusion::at<typename Arg0::arg_type::index_type>(m_metadata) = arg0;
            assign_metadatas(other_args...);
        }
#endif
    public:

#ifdef CXX11_ENABLED
        /** @brief variadic constructor
            construct the domain_type given an arbitrary number of placeholders to the non-temporary
            storages passed as arguments.

            USAGE EXAMPLE:
            \verbatim
            domain_type((p1=storage_1), (p2=storage_2), (p3=storage_3));
            \endverbatim
        */
        template <typename... StorageArgs, typename ... MetaStorageArgs>
        domain_type(storage<StorageArgs>... args, MetaStorageArgs ... meta_args)
            : m_storage_pointers()
            , m_metadata()
            {
                assign_pointers(args...);
                assign_metadata(meta_args...);
            }
#endif

        /**@brief Constructor from boost::fusion::vector
         * @tparam RealStorage fusion::vector of pointers to storages sorted with increasing indices of the pplaceholders
         * @param real_storage The actual fusion::vector with the values
         TODO: when I have only one placeholder and C++11 enabled this constructor is erroneously picked
         */
        template <typename RealStorage, typename MetaStorage>
        explicit domain_type(RealStorage const & real_storage_, MetaStorage const& meta_storage_)
            : m_storage_pointers()
            , m_metadata()
            , m_original_metadata()
            {

// #ifndef NDEBUG
            // the following creates an empty storage (problems with its destruction)
//             std::cout << "These are the original placeholders and their storages" << std::endl;
//             gridtools::for_each<original_placeholders>(_debug::stdcoutstuff());
// #endif

            typedef boost::fusion::filter_view<arg_list,
                is_storage<boost::mpl::_1> > view_type;

            // // typedef boost::fusion::filter_view<metadata_list,
            // //     is_meta_storage<boost::mpl::_1> > meta_view_type;
            // typedef metadata_list meta_view_type;

            view_type fview(m_storage_pointers);
            // meta_view_type fview_meta(m_metadata);

            GRIDTOOLS_STATIC_ASSERT( boost::fusion::result_of::size<view_type>::type::value == boost::mpl::size<RealStorage>::type::value, "The number of arguments specified when constructing the domain_type is not the same as the number of placeholders to non-temporary storages.");

            // GRIDTOOLS_STATIC_ASSERT( boost::fusion::result_of::size<meta_view_type>::type::value == boost::mpl::size<MetaStorage>::type::value, "The number of meta storages specified when constructing the domain_type is not the same as the number of meta storage types in the template arugment of the domain_type.");

// #ifndef NDEBUG
            //the following creates an empty storage (problems with its destruction)
//             // std::cout << "These are the actual placeholders and their storages" << std::endl;
//             // gridtools::for_each<placeholders>(_debug::stdcoutstuff());
//             std::cout << "These are the real storages" << std::endl;
//             boost::fusion::for_each(real_storage, _debug::print_deref());
//             std::cout << "\nThese are the arg_list elems" << std::endl;
//             boost::fusion::for_each(arg_list(), _debug::print_deref());
//             std::cout << "\nThese are the m_storage_pointers elems" << std::endl;
//             boost::fusion::for_each(arg_list(), _debug::print_deref());
//             std::cout << "\nThese are the view " << boost::fusion::size(fview) << std::endl;
//             boost::fusion::for_each(fview, _debug::print_deref());
// #endif
            boost::fusion::copy(real_storage_, fview);
            boost::fusion::copy(meta_storage_, m_metadata);
            boost::fusion::copy(meta_storage_, m_original_metadata);
            // boost::fusion::copy(meta_storage_, fview_meta);

#ifdef __VERBOSE__
            std::cout << "\nThese are the view values" << boost::fusion::size(fview) << std::endl;
            boost::fusion::for_each(m_storage_pointers, _debug::print_pointer());
#endif

            view_type original_fview(m_original_pointers);
            boost::fusion::copy(real_storage_, original_fview);
            //meta_view_type original_meta_fview(m_original_metadata);
            //original_meta_fview=meta_storage_;
            //=meta_storage_;
        }


#ifdef __CUDACC__
        /** Copy constructor to be used when cloning to GPU
         *
         * @param The object to copy. Typically this will be *this
         */
        __device__
        explicit domain_type(domain_type const& other)
            : m_storage_pointers(other.m_storage_pointers)
            , m_original_pointers(other.m_original_pointers)
            , m_metadata(other.m_metadata)
            , m_original_metadata(other.m_original_metadata)
        { }
#endif

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
            // std::cout << Index::value << " -|-> "
            //           << (boost::fusion::at_c<Index::value>(m_storage_pointers))->name()
            //           << " "
            //           << (boost::fusion::at_c<Index::value>(m_storage_pointers))->m_dims[0]
            //           << "x"
            //           << (boost::fusion::at_c<Index::value>(m_storage_pointers))->m_dims[1]
            //           << "x"
            //           << (boost::fusion::at_c<Index::value>(m_storage_pointers))->m_dims[2]
            //           << ", "
            //           << (boost::fusion::at_c<Index::value>(m_storage_pointers))->strides[0]
            //           << "x"
            //           << (boost::fusion::at_c<Index::value>(m_storage_pointers))->strides[1]
            //           << "x"
            //           << (boost::fusion::at_c<Index::value>(m_storage_pointers))->strides[2]
            //           << ", "
            //           << std::endl;
        }

        /** @brief copy the pointers from the device to the host */
        void finalize_computation() {
            boost::fusion::for_each(m_original_pointers, call_d2h());
            gridtools::for_each<
                boost::mpl::range_c<int, 0, boost::mpl::size<arg_list>::value >
            > (copy_pointers_functor<arg_list, arg_list> (m_original_pointers, m_storage_pointers));
        }

    };

    template<typename domain>
    struct is_domain_type : boost::mpl::false_ {};

    template <typename Placeholders, typename MetaData>
    struct is_domain_type<domain_type<Placeholders, MetaData> > : boost::mpl::true_{};

} // namespace gridtools

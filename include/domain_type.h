#pragma once

#include "defs.h"
#include "gt_assert.h"

#include <stdio.h>
#include <boost/mpl/vector.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/fusion/include/push_back.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/fusion/include/transform.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/copy.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/mpl/find.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/fusion/include/nview.hpp>
#include <boost/fusion/include/zip_view.hpp>
#include <boost/fusion/view/filter_view.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/copy.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include "gt_for_each/for_each.hpp"
#include <boost/type_traits/remove_pointer.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/mpl/distance.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/mpl/contains.hpp>

#include <boost/type_traits/remove_pointer.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include "arg_type.h"
#include "storage.h"
#include "layout_map.h"
#include "domain_type_impl.h"

namespace gridtools {

    namespace gt_aux {
        template<typename Iterator, typename Bound>
        GT_FUNCTION
        static void assert_in_range(Iterator pos, std::pair<Bound, Bound> min_max)
        {
            assert(pos >= min_max.first);
            assert(pos < min_max.second);
        }
    }

    /**
     * @tparam Placeholders list of placeholders of type arg<I,T>
     */
    template <typename Placeholders>
    struct domain_type : public clonable_to_gpu<domain_type<Placeholders> > {
        typedef Placeholders original_placeholders;
    private:
        BOOST_STATIC_CONSTANT(int, len = boost::mpl::size<original_placeholders>::type::value);

        typedef typename boost::mpl::transform<original_placeholders,
                                               _impl::l_get_type
                                               >::type raw_storage_list;

        typedef typename boost::mpl::transform<original_placeholders,
                                               _impl::l_get_it_type
                                               >::type raw_iterators_list;

    public:
        typedef typename boost::mpl::transform<original_placeholders,
                                               _impl::l_get_index
                                               >::type raw_index_list;
        typedef boost::mpl::range_c<int,0,len> range_t;
    private:    
        typedef typename boost::mpl::fold<range_t,
                                          boost::mpl::vector<>,
                                          boost::mpl::push_back<
                                              boost::mpl::_1,
                                              boost::mpl::find<raw_index_list, boost::mpl::_2>
                                              >
                                          >::type iter_list;

    public:
        typedef typename boost::mpl::transform<iter_list,
                                               _impl::l_get_it_pos
                                               >::type index_list;
   
        typedef typename boost::mpl::fold<index_list,
                                          boost::mpl::vector<>,
                                          boost::mpl::push_back<
                                              boost::mpl::_1, 
                                              boost::mpl::at<raw_storage_list, boost::mpl::_2> 
                                              >
                                          >::type arg_list_mpl;

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
                                          boost::mpl::push_back<boost::mpl::_1, boost::mpl::at<raw_iterators_list, boost::mpl::_2> >
                                          >::type iterator_list_mpl;
    
    public:
        /**
         * Type of fusion::vector of pointers to storages as indicated in Placeholders
         */
        typedef typename boost::fusion::result_of::as_vector<arg_list_mpl>::type arg_list;
        /**
         * Type of fusion::vector of pointers to iterators as indicated in Placeholders
         */
        typedef typename boost::fusion::result_of::as_vector<iterator_list_mpl>::type iterator_list;
    
        /**
         * fusion::vector of pointers to storages
         */
        arg_list storage_pointers;

        /**
         * fusion::vector of pointers to storages before the updates needed before the start of the computation
         */
        arg_list original_pointers;

        /**
         * fusion::vector of iterators used to access storage_pointers
         */
        iterator_list iterators;

        /**
         * States if the temporaries have been setup so the computation can start
         */
        bool is_ready;

    private:
        // Using zip view to associate iterators and storage_pointers in a single object.
        // This is used to move iterators to coordinates relative to storage
        typedef boost::fusion::vector<iterator_list&, arg_list&> zip_vector_type;
        zip_vector_type zip_vector;
        typedef boost::fusion::zip_view<zip_vector_type> zipping;
    public:

        /**
         * @tparam RealStorage fusion::vector of pointers to storages sorted with increasing indices of the pplaceholders
         * @param real_storage The actual fusion::vector with the values
         */
        template <typename RealStorage>
        explicit domain_type(RealStorage const & real_storage)
            : storage_pointers()
            , iterators()
            , zip_vector(iterators, storage_pointers)
            , is_ready(false)
        {
            typedef boost::fusion::filter_view<arg_list, 
                boost::mpl::not_<is_temporary_storage<boost::mpl::_> > > view_type;

            view_type fview(storage_pointers);

            BOOST_MPL_ASSERT_MSG( (boost::fusion::result_of::size<view_type>::type::value == boost::mpl::size<RealStorage>::type::value), _NUMBER_OF_ARGS_SEEMS_WRONG_, (boost::fusion::result_of::size<view_type>) );

            boost::fusion::copy(real_storage, fview);

            view_type original_fview(original_pointers);
            boost::fusion::copy(real_storage, fview);
        }

#ifdef __CUDACC__
        /** Copy constructor to be used when cloning to GPU
         * 
         * @param The object to copy. Typically this will be *this
         */
        __device__
        explicit domain_type(domain_type const& other) 
            : storage_pointers(other.storage_pointers)
            , original_pointers(other.original_pointers)
            , iterators(other.iterators)
            , zip_vector(iterators, storage_pointers)
            , is_ready(other.is_ready) // should no matter
        { }
#endif

        GT_FUNCTION
        void info() {
            printf("domain_type: Storage pointers\n");
            boost::fusion::for_each(storage_pointers, _debug::print_domain_info());
            printf("domain_type: Iterators\n");
            boost::fusion::for_each(iterators, _debug::print_domain_info());
            printf("domain_type: Original pointers\n");
            boost::fusion::for_each(original_pointers, _debug::print_domain_info());
            printf("domain_type: End info\n");
        }

        template <typename Index>
        void storage_info() const {
            std::cout << Index::value << " -|-> "
                      << (boost::fusion::at_c<Index::value>(storage_pointers))->name()
                      << " "
                      << (boost::fusion::at_c<Index::value>(storage_pointers))->m_dims[0]
                      << "x"
                      << (boost::fusion::at_c<Index::value>(storage_pointers))->m_dims[1]
                      << "x"
                      << (boost::fusion::at_c<Index::value>(storage_pointers))->m_dims[2]
                      << ", "
                      << (boost::fusion::at_c<Index::value>(storage_pointers))->strides[0]
                      << "x"
                      << (boost::fusion::at_c<Index::value>(storage_pointers))->strides[1]
                      << "x"
                      << (boost::fusion::at_c<Index::value>(storage_pointers))->strides[2]
                      << ", "
                      << std::endl;
        }

        ~domain_type() {
            typedef boost::fusion::filter_view<arg_list, 
                is_temporary_storage<boost::mpl::_> > tmp_view_type;
            tmp_view_type fview(storage_pointers);
            //boost::fusion::for_each(fview, _impl::delete_tmps());
        }

        /**
           This function calls h2d_update on all storages, in order to
           get the data prepared in the case of GPU execution.

           Returns 0 (GT_NO_ERRORS) on success
        */
        int setup_computation() {
            if (is_ready) {
#ifndef NDEBUG
                printf("Setup computation\n");
#endif
                boost::fusion::copy(storage_pointers, original_pointers);

                boost::fusion::for_each(storage_pointers, _impl::update_pointer());
#ifndef NDEBUG
                printf("POINTERS\n");
                boost::fusion::for_each(storage_pointers, _debug::print_pointer());
                printf("ORIGINAL\n");
                boost::fusion::for_each(original_pointers, _debug::print_pointer());
#endif
            } else {
#ifndef NDEBUG
                printf("Setup computation FAILED\n");
#endif
                return GT_ERROR_NO_TEMPS;
            }

            return GT_NO_ERRORS;
        }

        template <typename T>
        GT_FUNCTION
        typename boost::remove_pointer<typename boost::fusion::result_of::value_at<arg_list, typename T::index_type>::type>::type::value_type&
        operator[](T const &) {
            return *(boost::fusion::at<typename T::index_type>(iterators));
        }

        /**
         * Function to access the data pointed to a specific iterator
         * 
         * @tparam Index of the iterator in the iterator list
         * 
         * @return Reference to the value pointed to Ith iterator
         */
        template <typename I>
        GT_FUNCTION
        typename boost::remove_pointer<typename boost::fusion::result_of::value_at<arg_list, I>::type>::type::value_type&
        direct() const {
            #ifndef NDEBUG
            gt_aux::assert_in_range(
                boost::fusion::at<I>(iterators),
                std::make_pair(boost::fusion::at<I>(storage_pointers)->min_addr(),
                               boost::fusion::at<I>(storage_pointers)->max_addr()));
            #endif

            return *(boost::fusion::at<I>(iterators));
        }

        /**
         * Move all iterators of storages to (i,j,k) coordinates
         */
        GT_FUNCTION
        void move_to(int i, int j, int k) const {
            boost::fusion::for_each(zipping(zip_vector), _impl::moveto_functor(i,j,k));
            // Simpler code:
            // for (int l = 0; l < len; ++l) {
            //     iterators[l] = &( (*(storage_pointers[l]))(i,j,k) );
            //     assert(iterators[l] >= storage_pointers[l]->min_addr());
            //     assert(iterators[l] < storage_pointers[l]->max_addr());
            // }
        }

        /**
         * Move all iterators one position along one direction
         * 
         * @\tparam DIR index of coordinate to increment by one
         */
        template <int DIR>
        GT_FUNCTION
        void increment_along() const {
            boost::fusion::for_each(zipping(zip_vector), _impl::increment_functor<DIR>());
            // Simpler code:
            // for (int l = 0; l < len; ++l) {
            //     iterators[l] += (*(storage_pointers[l])).template stride_along<DIR>();
            // }
        }

    };

} // namespace gridtools

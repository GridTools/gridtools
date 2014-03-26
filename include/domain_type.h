#pragma once

#include "defs.h"
#include "gt_assert.h"

#include <boost/mpl/vector.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/fusion/include/push_back.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/fusion/include/transform.hpp>
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
    /**
     * @tparam t_placeholders list of placeholders of type arg<I,T>
     */
    template <typename t_placeholders>
    struct domain_type : public clonable_to_gpu<domain_type<t_placeholders> > {
        typedef t_placeholders original_placeholders;
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
         * Type of fusion::vector of pointers to storages as indicated in t_placeholders
         */
        typedef typename boost::fusion::result_of::as_vector<arg_list_mpl>::type arg_list;
        /**
         * Type of fusion::vector of pointers to iterators as indicated in t_placeholders
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
         * @tparam t_real_storage fusion::vector of pointers to storages sorted with increasing indices of the pplaceholders
         * @param real_storage The actual fusion::vector with the values
         */
        template <typename t_real_storage>
        explicit domain_type(t_real_storage const & real_storage)
            : storage_pointers()
            , iterators()
            , zip_vector(iterators, storage_pointers)
            , is_ready(false)
        {
            typedef boost::fusion::filter_view<arg_list, 
                boost::mpl::not_<is_temporary_storage<boost::mpl::_> > > view_type;

            view_type fview(storage_pointers);

            BOOST_MPL_ASSERT_MSG( (boost::fusion::result_of::size<view_type>::type::value == boost::mpl::size<t_real_storage>::type::value), _NUMBER_OF_ARGS_SEEMS_WRONG_, (boost::fusion::result_of::size<view_type>) );

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
            printf("domain_type: Original pointers\n");
            boost::fusion::for_each(original_pointers, _debug::print_domain_info());
            printf("domain_type: End info\n");
        }

        template <typename t_index>
        void storage_info() const {
            std::cout << t_index::value << " -|-> "
                      << (boost::fusion::template at_c<t_index::value>(storage_pointers))->name()
                      << std::endl;
        }

        ~domain_type() {
            typedef boost::fusion::filter_view<arg_list, 
                is_temporary_storage<boost::mpl::_> > tmp_view_type;
            tmp_view_type fview(storage_pointers);
            boost::fusion::for_each(fview, _impl::delete_tmps());

        }

        /**
         * This function is to be called by intermediate representation or back-end
         * 
         * @tparam t_mss_type The multistage stencil type as passed to the back-end
         * @tparam t_range_sizes mpl::vector with the sizes of the extents of the 
         *         access for each functor listed as linear_esf in t_mss_type
         * @tparam t_back_end This is not currently used and may be dropped in future
         * 
         * @param tileI Tile size in the first dimension as used by the back-end
         * @param tileJ Tile size in the second dimension as used by the back-end
         * @param tileK Tile size in the third dimension as used by the back-end
         */
        template <typename t_mss_type, typename t_range_sizes>
        void prepare_temporaries(int tileI, int tileJ, int tileK) {
#ifndef NDEBUG
            std::cout << "Prepare ARGUMENTS" << std::endl;
#endif

            // Got to find temporary indices
            typedef typename boost::mpl::fold<placeholders,
                boost::mpl::vector<>,
                boost::mpl::if_<
                   is_plchldr_to_temp<boost::mpl::_2>,
                       boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2 >,
                       boost::mpl::_1>
                >::type list_of_temporaries;
        
#ifndef NDEBUG
            std::cout << "BEGIN TMPS" << std::endl;
            for_each<list_of_temporaries>(_debug::print_index());
            std::cout << "END TMPS" << std::endl;
#endif
        
            // Compute a vector of vectors of temp indices of temporaris initialized by each functor
            typedef typename boost::mpl::fold<typename t_mss_type::linear_esf,
                    boost::mpl::vector<>,
                    boost::mpl::push_back<boost::mpl::_1, _impl::get_temps_per_functor<boost::mpl::_2> >
                    >::type temps_per_functor;

             typedef typename boost::mpl::transform<
                list_of_temporaries,
                _impl::associate_ranges<temps_per_functor, t_range_sizes>
                >::type list_of_ranges;

#ifndef NDEBUG
            std::cout << "BEGIN TMPS/F" << std::endl;
            for_each<temps_per_functor>(_debug::print_tmps());
            std::cout << "END TMPS/F" << std::endl;

            std::cout << "BEGIN RANGES/F" << std::endl;
            for_each<list_of_ranges>(_debug::print_ranges());
            std::cout << "END RANGES/F" << std::endl;

            std::cout << "BEGIN Fs" << std::endl;
            for_each<typename t_mss_type::linear_esf>(_debug::print_ranges());
            std::cout << "END Fs" << std::endl;
#endif
        
            typedef boost::fusion::filter_view<arg_list, 
                is_temporary_storage<boost::mpl::_> > tmp_view_type;
            tmp_view_type fview(storage_pointers);

#ifndef NDEBUG
            std::cout << "BEGIN VIEW" << std::endl;
            boost::fusion::for_each(fview, _debug::print_view());
            std::cout << "END VIEW" << std::endl;
#endif
        
            list_of_ranges lor;
            typedef boost::fusion::vector<tmp_view_type&, list_of_ranges const&> zipper;
            zipper zzip(fview, lor);
            boost::fusion::zip_view<zipper> zip(zzip); 
            boost::fusion::for_each(zip, _impl::instantiate_tmps(tileI, tileJ, tileK));

#ifndef NDEBUG
            std::cout << "BEGIN VIEW DOPO" << std::endl;
            boost::fusion::for_each(fview, _debug::print_view_());
            std::cout << "END VIEW DOPO" << std::endl;
#endif        

            is_ready = true;
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
                original_pointers = storage_pointers;

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

        /**
           This function calls d2h_update on all storages, in order to
           get the data back to the host after a computation.
        */
        void finalize_computation() {
            boost::fusion::for_each(original_pointers, _impl::call_d2h());
            boost::fusion::copy(original_pointers, storage_pointers);
        }

        template <typename T>
        GT_FUNCTION
        typename boost::remove_pointer<typename boost::fusion::result_of::value_at<arg_list, typename T::index_type>::type>::type::value_type&
        operator[](T const &) {
            return *(boost::fusion::template at<typename T::index_type>(iterators));
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
            assert((boost::fusion::template at<I>(iterators) >= boost::fusion::template at<I>(storage_pointers)->min_addr()));
            assert((boost::fusion::template at<I>(iterators) < boost::fusion::template at<I>(storage_pointers)->max_addr()));

            return *(boost::fusion::template at<I>(iterators));
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

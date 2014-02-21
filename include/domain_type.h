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

namespace gridtools {
    namespace _debug {
    
        struct print_index {
            template <typename T>
            void operator()(T& ) const {
                std::cout << " *" << T() << ", " << T::index_type::value << " * " << std::endl;
            }
        };

        struct print_tmps {
            template <typename T>
            void operator()(T& ) const {
                for_each<T>(print_index());
                std::cout << " ---" << std::endl;
            }
        };

        struct print_ranges {
            template <typename T>
            void operator()(T& ) const {
                std::cout << T() << std::endl;
            }
        };

        struct print_view {
            template <typename T>
            void operator()(T& t) const {
                // int a = T();
                boost::remove_pointer<T>::type::text();
            }
        };

        struct print_view_ {
            template <typename T>
            void operator()(T& t) const {
                // int a = T();
                t->info();
            }
        };
    } // namespace _debug

    namespace _impl {
        struct l_get_type {
            template <typename U>
            struct apply {
                typedef typename U::storage_type* type;
            };
        };

        struct l_get_index {
            template <typename U>
            struct apply {
                typedef typename U::index_type type;
            };
        };

        struct l_get_it_type {
            template <typename U>
            struct apply {
                typedef typename U::storage_type::iterator_type type;
            };
        };

        struct update_pointer {
            template <typename storage_t>
            GT_FUNCTION
            void operator()(storage_t* s) const {}
            
#ifdef __CUDACC__
            template <typename T, typename U, bool B>
            void operator()(cuda_storage<T,U,B> *& s) const {
                if (s) {
#ifndef NDEBUG
                    std::cout << "UPDATING " 
                              << std::hex << s->gpu_object_ptr 
                              << " " << s
                              << " " << sizeof(cuda_storage<T,U,B>)
                              << std::dec << std::endl;
#endif
                    // s->data.update_gpu();
                    // s->clone_to_gpu();
                    s = s->gpu_object_ptr;
                }
            }
#endif
        };
        
        struct moveto_functor {
            int i,j,k;
            GT_FUNCTION
            moveto_functor(int i, int j, int k) 
                : i(i)
                , j(j)
                , k(k)
            {}

            template <typename t_zip_elem>
            GT_FUNCTION
            void operator()(t_zip_elem const &a) const {
                boost::fusion::at<boost::mpl::int_<0> >(a) = &( (*(boost::fusion::at<boost::mpl::int_<1> >(a)))(i,j,k) );
            }
        };

        template <int DIR>
        struct increment_functor {
            template <typename t_zip_elem>
            GT_FUNCTION
            void operator()(t_zip_elem const &a) const {
                // Simpler code:
                // iterators[l] += (*(args[l])).template stride_along<DIR>();
                boost::fusion::at_c<0>(a) += (*(boost::fusion::at_c<1>(a))).template stride_along<DIR>();
            }
        };

        template <typename t_esf>
        struct is_written_temp {
            template <typename index>
            struct apply {
                typedef typename boost::mpl::if_<
                    is_plchldr_to_temp<typename boost::mpl::at<typename t_esf::args, index>::type>,
                    typename boost::mpl::if_<
                        boost::is_const<typename boost::mpl::at<typename t_esf::esf_function::arg_list, index>::type>,
                        typename boost::false_type,
                        typename boost::true_type
                        >::type,
                            typename boost::false_type
                            >::type type;
            };
        };

        template <typename t_esf>
        struct get_it {
            template <typename index>
            struct apply {
                typedef typename boost::mpl::at<typename t_esf::args, index>::type type;
            };
        };

        template <typename t_esf_f>
        struct get_temps_per_functor {
            typedef boost::mpl::range_c<int, 0, boost::mpl::size<typename t_esf_f::args>::type::value> range;
            typedef typename boost::mpl::fold<
                range,
                boost::mpl::vector<>,
                typename boost::mpl::if_<
                    typename is_written_temp<t_esf_f>::template apply<boost::mpl::_2>,
                    boost::mpl::push_back<
                        boost::mpl::_1, 
                        typename _impl::get_it<t_esf_f>::template apply<boost::mpl::_2> >,
                   boost::mpl::_1
               >
            >::type type;
        };

        template <typename temps_per_functor, typename t_range_sizes>
        struct associate_ranges {

            template <typename t_temp>
            struct is_temp_there {
                template <typename t_temps_in_esf>
                struct apply {
                    typedef typename boost::mpl::contains<
                        t_temps_in_esf,
                        t_temp >::type type;
                };
            };
        
            template <typename t_temp>
            struct apply {

                typedef typename boost::mpl::find_if<
                    temps_per_functor,
                    typename is_temp_there<t_temp>::template apply<boost::mpl::_> >::type iter;

                BOOST_MPL_ASSERT_MSG( ( boost::mpl::not_<typename boost::is_same<iter, typename boost::mpl::end<temps_per_functor>::type >::type >::type::value ) ,WHATTTTTTTTT_, (iter) );

                typedef typename boost::mpl::at<t_range_sizes, typename iter::pos>::type type;               
            };
        };
    
        template <typename t_back_end>
        struct instantiate_tmps {
            int tileI;
            int tileJ;
            int tileK;
        
            GT_FUNCTION
            instantiate_tmps(int tileI, int tileJ, int tileK)
                : tileI(tileI)
                , tileJ(tileJ)
                , tileK(tileK)
            {}
        
            // elem_type: an element in the data field place-holders list
            template <typename elem_type>
            GT_FUNCTION
            void operator()(elem_type  e) const {
#ifndef __CUDA_ARCH__
                typedef typename boost::fusion::result_of::value_at<elem_type, boost::mpl::int_<1> >::type range_type;
                typedef typename boost::remove_pointer<typename boost::remove_reference<typename boost::fusion::result_of::value_at<elem_type, boost::mpl::int_<0> >::type>::type>::type storage_type;

                boost::fusion::at_c<0>(e) = new storage_type(-range_type::iminus::value+range_type::iplus::value+tileI,
                                                             -range_type::jminus::value+range_type::jplus::value+tileJ,
                                                             tileK);
#endif
            }
        };

        struct delete_tmps {
            template <typename t_elem>
            GT_FUNCTION
            void operator()(t_elem & elem) const {
#ifndef __CUDA_ARCH__
                delete elem;
#endif
            }
        };

        struct call_h2d {
            template <typename t_arg>
            GT_FUNCTION
            void operator()(t_arg * arg) const {
#ifndef __CUDA_ARCH__
                arg->h2d_update();
#endif
            }
        };

        struct call_d2h {
            template <typename t_arg>
            GT_FUNCTION
            void operator()(t_arg * arg) const {
#ifndef __CUDA_ARCH__
                arg->d2h_update();
#endif
            }
        };

    } // namespace _impl

    /**
     * @tparam t_placeholders list of placeholders of type arg<I,T>
     */
    template <typename t_placeholders>
    struct domain_type : public clonable_to_gpu<domain_type<t_placeholders> > {
        typedef t_placeholders placeholders;
    private:
        BOOST_STATIC_CONSTANT(int, len = boost::mpl::size<placeholders>::type::value);

        typedef typename boost::mpl::transform<placeholders,
                                               _impl::l_get_type
                                               >::type raw_storage_list;

        typedef typename boost::mpl::transform<placeholders,
                                               _impl::l_get_it_type
                                               >::type raw_iterators_list;

        typedef typename boost::mpl::transform<placeholders,
                                               _impl::l_get_index
                                               >::type raw_index_list;
    
        typedef typename boost::mpl::fold<raw_index_list,
                                          boost::mpl::vector<>,
                                          typename boost::mpl::push_back<boost::mpl::_1, boost::mpl::at<raw_storage_list, boost::mpl::_2> >
                                          >::type arg_list_mpl;

        typedef typename boost::mpl::fold<raw_index_list,
                                          boost::mpl::vector<>,
                                          typename boost::mpl::push_back<boost::mpl::_1, boost::mpl::at<raw_iterators_list, boost::mpl::_2> >
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
         * fusion::vector of iterators used to access storage_pointers
         */
        iterator_list iterators;

    private:
        // Using zip view to associate iterators and storage_pointers in a single object.
        // This is used to move iterators to coordinates relative to storage
        typedef typename boost::fusion::vector<iterator_list&, arg_list&> zip_vector_type;
        zip_vector_type zip_vector;
        typedef typename boost::fusion::zip_view<zip_vector_type> zipping;
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
        {
            typedef typename boost::fusion::filter_view<arg_list, 
                boost::mpl::not_<is_temporary_storage<boost::mpl::_> > > view_type;

            view_type fview(storage_pointers);

            BOOST_MPL_ASSERT_MSG( (boost::fusion::result_of::size<view_type>::type::value == boost::mpl::size<t_real_storage>::type::value), _NUMBER_OF_ARGS_SEEMS_WRONG_, (boost::fusion::result_of::size<view_type>) );

            boost::fusion::copy(real_storage, fview);
            boost::fusion::for_each(fview, _impl::update_pointer());

        }

#ifdef __CUDACC__
        /** Copy constructor to be used when cloning to GPU
         * 
         * @param The object to copy. Typically this will be *this
         */
        __device__
        explicit domain_type(domain_type const& other)
            : storage_pointers(other.storage_pointers)
            , iterators(other.iterators)
            , zip_vector(iterators, storage_pointers)
        { }
#endif

        ~domain_type() {
            typedef typename boost::fusion::filter_view<arg_list, 
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
        template <typename t_mss_type, typename t_range_sizes, typename t_back_end>
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
                    boost::mpl::push_back<boost::mpl::_1, typename _impl::get_temps_per_functor<boost::mpl::_2> >
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
        
            typedef typename boost::fusion::filter_view<arg_list, 
                is_temporary_storage<boost::mpl::_> > tmp_view_type;
            tmp_view_type fview(storage_pointers);

#ifndef NDEBUG
            std::cout << "BEGIN VIEW" << std::endl;
            boost::fusion::for_each(fview, _debug::print_view());
            std::cout << "END VIEW" << std::endl;
#endif
        
            list_of_ranges lor;
            typedef typename boost::fusion::vector<tmp_view_type&, list_of_ranges const&> zipper;
            zipper zzip(fview, lor);
            boost::fusion::zip_view<zipper> zip(zzip); 
            boost::fusion::for_each(zip, _impl::instantiate_tmps< t_back_end >(tileI, tileJ, tileK));

#ifndef NDEBUG
            std::cout << "BEGIN VIEW DOPO" << std::endl;
            boost::fusion::for_each(fview, _debug::print_view_());
            std::cout << "END VIEW DOPO" << std::endl;
#endif        
        }

        /**
           This function calls h2d_update on all storages, in order to
           get the data prepared in the case of GPU execution.
        */
        void setup_computation() {
            boost::fusion::for_each(storage_pointers, _impl::call_h2d());
        }

        /**
           This function calls d2h_update on all storages, in order to
           get the data back to the host after a computation.
        */
        void finalize_computation() {
            boost::fusion::for_each(storage_pointers, _impl::call_d2h());
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

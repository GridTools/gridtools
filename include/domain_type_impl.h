#pragma once

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

        struct print_pointer {
            template <typename storage_t>
            GT_FUNCTION_WARNING
            void operator()(storage_t* s) const {
                printf("CIAOOO TATATA %x\n", s);
            }
            
#ifdef __CUDACC__
            template <typename T, typename U, bool B
#ifndef NDEBUG
                      , typename type_tag
#endif
                      >
            GT_FUNCTION_WARNING
            void operator()(cuda_storage<T,U,B
#ifndef NDEBUG
                            , type_tag
#endif
                            > *& s) const {
                printf("CIAO POINTER %X\n", s);
            }
#endif
        };
        
        struct print_domain_info {
            template <typename storage_t>
            GT_FUNCTION
            void operator()(storage_t* s) const {
                printf("PTR %x\n", s);
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
            GT_FUNCTION_WARNING
            void operator()(storage_t* s) const {}
            
#ifdef __CUDACC__
            template <typename T, typename U, bool B
#ifndef NDEBUG
                      , typename type_tag
#endif
                      >
            GT_FUNCTION_WARNING
            void operator()(cuda_storage<T,U,B
#ifndef NDEBUG
                      , type_tag
#endif
                            > *& s) const {
                if (s) {
#ifndef NDEBUG
                    // std::cout << "UPDATING " 
                    //           << std::hex << s->gpu_object_ptr 
                    //           << " " << s
                    //           << " " << sizeof(cuda_storage<T,U,B>)
                    //           << std::dec << std::endl;
#endif
                    s->data.update_gpu();
                    s->clone_to_gpu();
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
#ifdef __CUDA_ARCH__
                printf("CIAOLLLL %X\n", &a);//, (boost::fusion::at<boost::mpl::int_<1> >(a)));
#endif
                //                (*(boost::fusion::at<boost::mpl::int_<1> >(a)))(i,j,k);
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

} // namespace gridtoold

#pragma once

#include <stdio.h>
#ifndef __CUDACC__
#include <boost/lexical_cast.hpp>
#endif

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
            template <typename StorageType>
            GT_FUNCTION_WARNING
            void operator()(StorageType* s) const {
                printf("CIAOOO TATATA %x\n", s);
            }

#ifdef __CUDACC__
            template <typename T, typename U, bool B
#ifndef NDEBUG
                      , typename TypeTag
#endif
                      >
            GT_FUNCTION_WARNING
            void operator()(cuda_storage<T,U,B
#ifndef NDEBUG
                            , TypeTag
#endif
                            > *& s) const {
                printf("CIAO POINTER %X\n", s);
            }
#endif
        };

        struct print_domain_info {
            template <typename StorageType>
            GT_FUNCTION
            void operator()(StorageType* s) const {
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

        struct l_get_it_pos {
            template <typename U>
            struct apply {
                typedef typename U::pos type;
            };
        };

        struct l_get_index {
            template <typename U>
            struct apply {
                typedef typename boost::mpl::integral_c<int, U::index_type::value> type;
            };
        };

        struct l_get_it_type {
            template <typename U>
            struct apply {
                typedef typename U::storage_type::iterator_type type;
            };
        };

        struct update_pointer {
            template <typename StorageType>
            GT_FUNCTION_WARNING
            void operator()(StorageType* s) const {}

#ifdef __CUDACC__
            template <typename T, typename U, bool B
#ifndef NDEBUG
                      , typename TypeTag
#endif
                      >
            GT_FUNCTION_WARNING
            void operator()(cuda_storage<T,U,B
#ifndef NDEBUG
                      , TypeTag
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

        struct call_h2d {
            template <typename Arg>
            GT_FUNCTION
            void operator()(Arg * arg) const {
#ifndef __CUDA_ARCH__
                arg->h2d_update();
#endif
            }
        };

        struct call_d2h {
            template <typename Arg>
            GT_FUNCTION
            void operator()(Arg * arg) const {
#ifndef __CUDA_ARCH__
                arg->d2h_update();
#endif
            }
        };

        struct moveto_functor {
            int i,j,k;
            GT_FUNCTION
            moveto_functor(int i, int j, int k)
                : i(i)
                , j(j)
                , k(k)
            {}

            template <typename ZipElem>
            GT_FUNCTION
            void operator()(ZipElem const &a) const {
#ifdef __CUDA_ARCH__
                printf("CIAOLLLL %X\n", &a);//, (boost::fusion::at<boost::mpl::int_<1> >(a)));
#endif
                //                (*(boost::fusion::at<boost::mpl::int_<1> >(a)))(i,j,k);
                boost::fusion::at<boost::mpl::int_<0> >(a) = &( (*(boost::fusion::at<boost::mpl::int_<1> >(a)))(i,j,k) );
            }
        };

        template <int DIR>
        struct increment_functor {
            template <typename ZipElem>
            GT_FUNCTION
            void operator()(ZipElem const &a) const {
                // Simpler code:
                // iterators[l] += (*(args[l])).template stride_along<DIR>();
                boost::fusion::at_c<0>(a) += (*(boost::fusion::at_c<1>(a))).template stride_along<DIR>();
            }
        };

        template <typename TempsPerFunctor, typename RangeSizes>
        struct associate_ranges {

            template <typename Temp>
            struct is_temp_there {
                template <typename TempsInEsf>
                struct apply {
                    typedef typename boost::mpl::contains<
                        TempsInEsf,
                        Temp >::type type;
                };
            };

            template <typename Temp>
            struct apply {

                typedef typename boost::mpl::find_if<
                    TempsPerFunctor,
                    typename is_temp_there<Temp>::template apply<boost::mpl::_> >::type iter;

                BOOST_MPL_ASSERT_MSG( ( boost::mpl::not_<typename boost::is_same<iter, typename boost::mpl::end<TempsPerFunctor>::type >::type >::type::value ) ,WHATTTTTTTTT_, (iter) );

                typedef typename boost::mpl::at<RangeSizes, typename iter::pos>::type type;
            };
        };

    } // namespace _impl

} // namespace gridtoold

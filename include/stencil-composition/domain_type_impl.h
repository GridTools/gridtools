#pragma once

#include <stdio.h>
#include <boost/type_traits/remove_pointer.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/insert.hpp>
#include <gt_for_each/for_each.hpp>

    template <typename RegularStorageType>
    struct no_storage_type_yet;

namespace gridtools {
    namespace _debug {

        struct stdcoutstuff {
            template <typename T>
            void operator()(T& ) const {
                std::cout << T() << " : " << typename T::storage_type() << std::endl;
            }
        };

        struct print_index {
            template <typename T>
            void operator()(T& ) const {
                std::cout << " *" << T() << " " << typename T::storage_type() << ", " << T::index_type::value << " * " << std::endl;
            }
        };

        struct print_tmps {
            template <typename T>
            void operator()(T& ) const {
                for_each<T>(print_index());
                std::cout << " --- " << std::endl;
            }
        };

        struct print_ranges {
            template <typename T>
            void operator()(T const&) const {
                std::cout << T() << std::endl;
            }
        };

        struct print_deref {
            template <typename T>
            void operator()(T* const&) const {
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


        struct print_domain_info {
            template <typename StorageType>
            GT_FUNCTION
            void operator()(StorageType* s) const {
                printf("PTR %x\n",  s);
            }
        };
    } // namespace _debug

    namespace _impl {
        struct l_get_type {
            template <typename U, typename Dummy = void>
            struct apply {
                typedef typename U::storage_type* type;
            };

            template <typename TheStorage, typename Dummy>
            struct apply<no_storage_type_yet<TheStorage>, Dummy> {
                typedef no_storage_type_yet<TheStorage>* type;
            };
        };

        struct l_get_it_pos {
            template <typename U>
            struct apply {
                typedef typename U::pos type;
            };
        };

        /**
           \brief returns the index chosen when the placeholder U was defined
        */
        struct l_get_index {
            template <typename U>
            struct apply {
#ifndef CXX11_ENABLED
                typedef typename static_uint< U::index_type::value >::type
#else
                typedef static_uint< U::index_type::value >
#endif
type;
            };
        };

        /**
           \brief returns the pointer to the storage for the specific domain placeholder U
        */
        struct l_get_it_type {
            template <typename U>
            struct apply {
                typedef typename U::storage_type::iterator_type type;
            };
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
            uint_t i,j,k;
            GT_FUNCTION
            moveto_functor(uint_t i, uint_t j, uint_t k)
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
                boost::fusion::at<static_int<0> >(a) = &( (*(boost::fusion::at<static_int<1> >(a)))(i,j,k) );
            }
        };

        template <ushort_t DIR>
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

	template <typename OriginalPlaceholders>
	struct compute_index_set{

	    /**
	     * \brief Get a sequence of the same type as original_placeholders, containing the indexes relative to the placehoolders
	     * note that the static const indexes are transformed into types using mpl::integral_c
	     */
	    typedef typename boost::mpl::transform<OriginalPlaceholders,
						   l_get_index
						   >::type raw_index_list;

	    /**@brief length of the index list eventually with duplicated indices */
	    static const uint_t len=boost::mpl::size<raw_index_list>::value;

	    /**
	       @brief filter out duplicates
	       check if the indexes are repeated (a common error is to define 2 types with the same index)
	     */
	    typedef typename boost::mpl::fold<raw_index_list,
					      boost::mpl::set<>,
					      boost::mpl::insert<boost::mpl::_1, boost::mpl::_2>
					      >::type index_set;
	};


    } // namespace _impl

} // namespace gridtoold

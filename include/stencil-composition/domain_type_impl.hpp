#pragma once

#include "accessor.hpp"

#include <gridtools.hpp>
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

        /**
         * @brief metafunction that computes the list of ranges associated to each functor.
         * It assumes the temporary is written only by one esf.
         * TODO This assumption is probably wrong?, a temporary could be written my multiple esf concatenated. The algorithm
         * we need to use here is find the maximum range associated to a temporary instead.
         * @tparam TempsPerFunctor vector of vectors containing the list of temporaries written per esf
         * @tparam RangeSizes ranges associated to each esf (i.e. due to read access patterns of later esf's)
         */
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

                GRIDTOOLS_STATIC_ASSERT(( boost::mpl::not_<typename boost::is_same<iter, typename boost::mpl::end<TempsPerFunctor>::type >::type >::type::value ) ,
                                      "Temporary not found in the list of temporaries" );

                typedef typename boost::mpl::at<RangeSizes, typename iter::pos>::type type;
            };
        };

        /**
         * @brief metafunction that computes the list of ranges associated to each functor.
         * It assumes the temporary is written only by one esf.
         * TODO This assumption is probably wrong?, a temporary could be written my multiple esf concatenated. The algorithm
         * we need to use here is find the maximum range associated to a temporary instead.
         * @tparam TempsPerFunctor vector of vectors containing the list of temporaries written per esf
         * @tparam RangeSizes ranges associated to each esf (i.e. due to read access patterns of later esf's)
         */
        template <typename TMap, typename Temp, typename TempsPerFunctor, typename RangeSizes>
        struct associate_ranges_map {
            template <typename TTemp>
            struct is_temp_there {
                template <typename TempsInEsf>
                struct apply {
                    typedef typename boost::mpl::contains<
                        TempsInEsf,
                        TTemp >::type type;
                };
            };

            typedef typename boost::mpl::find_if<
                TempsPerFunctor,
                typename is_temp_there<Temp>::template apply<boost::mpl::_>
                >::type iter;

            typedef typename boost::mpl::if_<
                boost::is_same<iter, typename boost::mpl::end<TempsPerFunctor>::type >,
                TMap,
                typename boost::mpl::insert<
                    TMap,
                    boost::mpl::pair<Temp, typename boost::mpl::at<RangeSizes, typename iter::pos>::type>
                    >::type
                >::type type;

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


        /** Metafunction.
         * Provide a list of placeholders to temporaries from a list os placeholders.
         */
        template <typename ListOfPlaceHolders>
        struct extract_temporaries {
            typedef typename boost::mpl::fold<ListOfPlaceHolders,
                                              boost::mpl::vector<>,
                                              boost::mpl::if_<
                                                  is_plchldr_to_temp<boost::mpl::_2>,
                                                  boost::mpl::push_back<boost::mpl::_1, boost::mpl::_2 >,
                                                  boost::mpl::_1>
                                              >::type type;
        };
    } // namespace _impl

} // namespace gridtoold

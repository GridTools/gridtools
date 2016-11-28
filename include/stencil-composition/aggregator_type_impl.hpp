/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

#include "accessor.hpp"
#include <iosfwd>

#include "arg.hpp"
#include <boost/fusion/include/at.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/type_traits/remove_pointer.hpp>
#include <gridtools.hpp>
#include <stdio.h>
template < typename RegularStorageType >
struct no_storage_type_yet;

namespace gridtools {
    namespace _debug {

        struct stdcoutstuff {
            template < typename T >
            void operator()(T &) const {
                std::cout << T() << " : " << typename T::storage_type() << std::endl;
            }
        };

        struct print_index {
            template < typename T >
            void operator()(T &) const {
                std::cout << " *" << T() << " " << typename T::storage_type() << ", " << T::index_t::value << " * "
                          << std::endl;
            }
        };

        struct print_tmps {
            template < typename T >
            void operator()(T &) const {
                boost::mpl::for_each< T >(print_index());
                std::cout << " --- " << std::endl;
            }
        };

        struct print_extents {
            template < typename T >
            void operator()(T const &) const {
                std::cout << T() << std::endl;
            }
        };

        struct print_deref {
            template < typename T >
            void operator()(T *const &) const {
                std::cout << T() << std::endl;
            }
        };

        struct print_view {
            template < typename T >
            void operator()(T &t) const {
                // int a = T();
                boost::remove_pointer< T >::type::text();
            }
        };

        struct print_view_ {
            std::ostream &out_s;
            print_view_(std::ostream &out_s) : out_s(out_s) {}

            template < typename T >
            void operator()(T &t) const {
                t->info(out_s);
            }
        };

        struct print_domain_info {
            template < typename StorageType >
            GT_FUNCTION void operator()(pointer< StorageType > s) const {
                printf("PTR %x\n", s);
            }
        };

        struct dt_print_pointer {
            template < typename PType >
            GT_FUNCTION_WARNING void operator()(PType const &s) const {
                printf("Pointer Value %x\n", s.get());
            }
        };

    } // namespace _debug

    template < typename Storage, uint_t Size >
    struct expandable_parameters;

    namespace _impl {
        struct l_get_type {
            template < typename U, typename Dummy = void >
            struct apply {
                typedef pointer< typename U::storage_type > type;
            };

            template < typename TheStorage, typename Dummy >
            struct apply< no_storage_type_yet< TheStorage >, Dummy > {
                typedef pointer< no_storage_type_yet< TheStorage > > type;
            };
        };

        /**
           \brief returns the index chosen when the placeholder U was defined
        */
        struct l_get_index {
            template < typename U >
            struct apply {
#ifndef CXX11_ENABLED
                typedef typename static_uint< U::index_t::value >::type
#else
                typedef static_uint< U::index_t::value >
#endif
                    type;
            };
        };

        /**
           \brief returns the pointer to the storage for the specific domain placeholder U
        */
        struct l_get_it_type {
            template < typename U >
            struct apply {
                GRIDTOOLS_STATIC_ASSERT((is_arg< U >::type::value), "wrong type");
                typedef typename U::iterator type;
            };
        };

        struct moveto_functor {
            uint_t i, j, k;
            GT_FUNCTION
            moveto_functor(uint_t i, uint_t j, uint_t k) : i(i), j(j), k(k) {}

            template < typename ZipElem >
            GT_FUNCTION void operator()(ZipElem const &a) const {
                boost::fusion::at< static_int< 0 > >(a) = &((*(boost::fusion::at< static_int< 1 > >(a)))(i, j, k));
            }
        };

        template < ushort_t DIR >
        struct increment_functor {
            template < typename ZipElem >
            GT_FUNCTION void operator()(ZipElem const &a) const {
                // Simpler code:
                // iterators[l] += (*(args[l])).template stride_along<DIR>();
                boost::fusion::at_c< 0 >(a) += (*(boost::fusion::at_c< 1 >(a))).template stride_along< DIR >();
            }
        };

        /**
         * @brief metafunction that computes the list of extents associated to each functor.
         * It assumes the temporary is written only by one esf.
         * TODO This assumption is probably wrong?, a temporary could be written my multiple esf concatenated. The
         * algorithm
         * we need to use here is find the maximum extent associated to a temporary instead.
         * @tparam TempsPerFunctor vector of vectors containing the list of temporaries written per esf
         * @tparam ExtendSizes extents associated to each esf (i.e. due to read access patterns of later esf's)
         */
        template < typename TempsPerFunctor, typename ExtendSizes >
        struct associate_extents {
            template < typename Temp >
            struct is_temp_there {
                template < typename TempsInEsf >
                struct apply {
                    typedef typename boost::mpl::contains< TempsInEsf, Temp >::type type;
                };
            };

            template < typename Temp >
            struct apply {

                typedef typename boost::mpl::find_if< TempsPerFunctor,
                    typename is_temp_there< Temp >::template apply< boost::mpl::_ > >::type iter;

                GRIDTOOLS_STATIC_ASSERT(
                    (boost::mpl::not_< typename boost::is_same< iter,
                            typename boost::mpl::end< TempsPerFunctor >::type >::type >::type::value),
                    "Temporary not found in the list of temporaries");

                typedef typename boost::mpl::at< ExtendSizes, typename iter::pos >::type type;
            };
        };

        namespace {
            template < typename T1, typename T2 >
            struct matching {
                typedef typename boost::is_same< T1, T2 >::type type;
            };

            // // specialization for base_storage
            // template < typename T1, typename T2, uint_t Size, ushort_t ID >
            // struct matching< arg< ID, std::vector< pointer< no_storage_type_yet< T1 > > > >,
            //                  arg< ID, no_storage_type_yet< expandable_parameters< T2, Size > > > > {
            //     typedef typename boost::is_same< T1, T2 >::type type;
            // };

            // specialization for storage
            template < typename T1, typename T2, uint_t Size, ushort_t ID >
            struct matching< arg< ID, std::vector< pointer< no_storage_type_yet< storage< T1 > > > > >,
                arg< ID, no_storage_type_yet< storage< expandable_parameters< T2, Size > > > > > {
                typedef typename boost::is_same< T1, T2 >::type type;
            };

            template < typename T1, typename T2 >
            struct contains {
                typedef typename boost::mpl::fold< T1,
                    boost::mpl::false_,
                    boost::mpl::or_< boost::mpl::_1, matching< boost::mpl::_2, T2 > > >::type type;
            };
        } // namespace

        /**
         * @brief metafunction that computes the list of extents associated to each functor.
         * It assumes the temporary is written only by one esf.
         * TODO This assumption is probably wrong?, a temporary could be written my multiple esf concatenated. The
         * algorithm
         * we need to use here is find the maximum extent associated to a temporary instead.
         * @tparam TempsPerFunctor vector of vectors containing the list of temporaries written per esf
         * @tparam ExtendSizes extents associated to each esf (i.e. due to read access patterns of later esf's)
         */
        template < typename TMap, typename Temp, typename TempsPerFunctor, typename ExtendSizes >
        struct associate_extents_map {
            template < typename TTemp >
            struct is_temp_there {
                template < typename TempsInEsf >
                struct apply {
                    typedef typename contains< TempsInEsf, TTemp >::type type;
                };
            };

            typedef typename boost::mpl::find_if< TempsPerFunctor,
                typename is_temp_there< Temp >::template apply< boost::mpl::_ > >::type iter;

            typedef typename boost::mpl::if_<
                typename boost::is_same< iter, typename boost::mpl::end< TempsPerFunctor >::type >::type,
                TMap,
                typename boost::mpl::insert< TMap,
                    boost::mpl::pair< Temp, typename boost::mpl::at< ExtendSizes, typename iter::pos >::type > >::
                    type >::type type;
        };

        template < typename OriginalPlaceholders >
        struct compute_index_set {

            /**
             * \brief Get a sequence of the same type as original_placeholders, containing the indexes relative to the
             * placehoolders
             * note that the static const indexes are transformed into types using mpl::integral_c
             */
            typedef typename boost::mpl::transform< OriginalPlaceholders, l_get_index >::type raw_index_list;

            /**@brief length of the index list eventually with duplicated indices */
            static const uint_t len = boost::mpl::size< raw_index_list >::value;

            /**
               @brief filter out duplicates
               check if the indexes are repeated (a common error is to define 2 types with the same index)
            */
            typedef typename boost::mpl::fold< raw_index_list,
                boost::mpl::set<>,
                boost::mpl::insert< boost::mpl::_1, boost::mpl::_2 > >::type index_set;
        };

        /** Metafunction.
         * Provide a list of placeholders to temporaries from a list os placeholders.
         */
        template < typename ListOfPlaceHolders >
        struct extract_temporaries {
            typedef typename boost::mpl::fold< ListOfPlaceHolders,
                boost::mpl::vector0<>,
                boost::mpl::if_< is_plchldr_to_temp< boost::mpl::_2 >,
                                                   boost::mpl::push_back< boost::mpl::_1, boost::mpl::_2 >,
                                                   boost::mpl::_1 > >::type type;
        };
    } // namespace _impl

} // namespace gridtoold

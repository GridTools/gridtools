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
/*
  @file
  This file provides functionality for a iterate domain remapper that intercepts calls to iterate domain
  and remap the arguments to the actual positions in the iterate domain
*/

#pragma once
#include "../iterate_domain_metafunctions.hpp"
#include "stencil-composition/accessor.hpp"
#include "../iterate_domain_fwd.hpp"

namespace gridtools {

    namespace icgrid {

        namespace _aux {
            template < typename T >
            struct iterate_domain_remapper_base_iterate_domain;

            template < typename IterateDomain,
                typename EsfArgsMap,
                typename EsfLocationType,
                uint_t Color,
                template < typename, typename, typename, uint_t > class Impl >
            struct iterate_domain_remapper_base_iterate_domain<
                Impl< IterateDomain, EsfArgsMap, EsfLocationType, Color > > {
                typedef IterateDomain type;
            };

            template < typename T >
            struct iterate_domain_remapper_base_color;

            template < typename IterateDomain,
                typename EsfArgsMap,
                typename EsfLocationType,
                uint_t Color,
                template < typename, typename, typename, uint_t > class Impl >
            struct iterate_domain_remapper_base_color< Impl< IterateDomain, EsfArgsMap, EsfLocationType, Color > > {
                typedef static_uint< Color > type;
            };

            template < typename T >
            struct iterate_domain_remapper_base_esf_location_type;

            template < typename IterateDomain,
                typename EsfArgsMap,
                typename EsfLocationType,
                uint_t Color,
                template < typename, typename, typename, uint_t > class Impl >
            struct iterate_domain_remapper_base_esf_location_type<
                Impl< IterateDomain, EsfArgsMap, EsfLocationType, Color > > {
                typedef EsfLocationType type;
            };

            template < typename T >
            struct iterate_domain_remapper_base_esf_args_map;

            template < typename IterateDomain,
                typename EsfArgsMap,
                typename EsfLocationType,
                uint_t Color,
                template < typename, typename, typename, uint_t > class Impl >
            struct iterate_domain_remapper_base_esf_args_map<
                Impl< IterateDomain, EsfArgsMap, EsfLocationType, Color > > {
                typedef EsfArgsMap type;
            };
        } // namespace _aux

        /**
         * @class iterate_domain_remapper_base
         * base class of an iterate_domain_remapper that intercepts the calls to evaluate the value of an arguments
         * from the iterate domain, and redirect the arg specified by user to the actual position of the arg in the
         * iterate domain
         * @param IterateDomainEvaluatorImpl implementer class of the CRTP
         */
        template < typename IterateDomainEvaluatorImpl >
        class iterate_domain_remapper_base {
            DISALLOW_COPY_AND_ASSIGN(iterate_domain_remapper_base);

          public:
            typedef typename _aux::iterate_domain_remapper_base_iterate_domain< IterateDomainEvaluatorImpl >::type
                iterate_domain_t;
            typedef typename _aux::iterate_domain_remapper_base_color< IterateDomainEvaluatorImpl >::type color_t;

            typedef typename _aux::iterate_domain_remapper_base_esf_location_type< IterateDomainEvaluatorImpl >::type
                esf_location_type_t;

          protected:
            const iterate_domain_t &m_iterate_domain;

          public:
            typedef typename _aux::iterate_domain_remapper_base_esf_args_map< IterateDomainEvaluatorImpl >::type
                esf_args_map_t;

            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< iterate_domain_t >::value), "Internal Error: wrong type");
            typedef typename iterate_domain_t::esf_args_t esf_args_t;

#ifdef CXX11_ENABLED
            template < typename Accessor >
            using accessor_return_type = typename iterate_domain_t::template accessor_return_type<
                typename remap_accessor_type< Accessor, esf_args_map_t >::type >;
#else
            template < typename Accessor >
            struct accessor_return_type {
                typedef typename iterate_domain_t::template accessor_return_type<
                    typename remap_accessor_type< Accessor, esf_args_map_t >::type >::type type;
            };
#endif

            GT_FUNCTION
            array< uint_t, 4 > const &position() const { return m_iterate_domain.position(); }

            GT_FUNCTION
            explicit iterate_domain_remapper_base(const iterate_domain_t &iterate_domain)
                : m_iterate_domain(iterate_domain) {}

            GT_FUNCTION
            iterate_domain_t const &get() const { return m_iterate_domain; }

            /** shifting the IDs of the placeholders and forwarding to the iterate_domain () operator*/

            template < typename Accessor >
            GT_FUNCTION auto operator()(Accessor const &arg) const
                -> decltype(m_iterate_domain(typename remap_accessor_type< Accessor, esf_args_map_t >::type(arg)))

            {
                typedef typename remap_accessor_type< Accessor, esf_args_map_t >::type remap_accessor_t;
                return m_iterate_domain(remap_accessor_t(arg));
            }

            template < typename ValueType, typename LocationTypeT, typename Reduction, typename... Accessors >
            GT_FUNCTION ValueType operator()(
                on_neighbors< ValueType, LocationTypeT, Reduction, Accessors... > onneighbors) const {

                typedef on_neighbors_impl< ValueType,
                    color_t,
                    LocationTypeT,
                    Reduction,
                    typename remap_accessor_type< Accessors, esf_args_map_t >::type... > remap_accessor_t;
                return m_iterate_domain(esf_location_type_t(), remap_accessor_t(onneighbors));
            }
        };

        /**
         * @class iterate_domain_remapper
         * default iterate domain remapper when positional information is not required
         * @param IterateDomain iterate domain
         * @param EsfArgsMap map from ESF arguments to iterate domain position of args.
         */
        template < typename IterateDomain, typename EsfArgsMap, typename EsfLocationType, uint_t Color >
        class iterate_domain_remapper
            : public iterate_domain_remapper_base<
                  iterate_domain_remapper< IterateDomain, EsfArgsMap, EsfLocationType, Color > > // CRTP
        {
            DISALLOW_COPY_AND_ASSIGN(iterate_domain_remapper);

          public:
            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< IterateDomain >::value), "Internal Error: wrong type");
            typedef iterate_domain_remapper_base<
                iterate_domain_remapper< IterateDomain, EsfArgsMap, EsfLocationType, Color > > super;

            GT_FUNCTION
            explicit iterate_domain_remapper(const IterateDomain &iterate_domain) : super(iterate_domain) {}
        };
    }

    /** Metafunction to query an iterate domain if it's positional. Specialization for
        iterate_domain_remapper
    */
    template < typename T, typename U, typename L, uint_t C >
    struct is_positional_iterate_domain< icgrid::iterate_domain_remapper< T, U, L, C > > : boost::false_type {};

    /** Metafunction to query a type is an iterate domain.
    */
    template < typename T, typename U, typename L, uint_t C >
    struct is_iterate_domain< icgrid::iterate_domain_remapper< T, U, L, C > > : boost::true_type {};

    /**
     * @struct get_iterate_domain_remapper
     * metafunction that computes the iterate_domain_remapper from the iterate domain type
     */
    template < typename IterateDomain, typename EsfArgsMap, typename EsfLocationType, uint_t Color >
    struct get_iterate_domain_remapper {
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< IterateDomain >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_location_type< EsfLocationType >::value), "Internal Error: wrong type");

        typedef icgrid::iterate_domain_remapper< IterateDomain, EsfArgsMap, EsfLocationType, Color > type;
    };

    /**
     * @struct get_trivial_iterate_domain_remapper
     * metafunction that computes a trivial iterate_domain_remapper where all the accessors are mapped to themselves
     */
    template < typename IterateDomain, typename Esf, typename Color >
    struct get_trivial_iterate_domain_remapper {
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< IterateDomain >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< Esf >::value), "Internal Error: wrong type");

        template < typename Map, typename Item >
        struct insert_ {
            typedef typename boost::mpl::insert< Map,
                boost::mpl::pair< boost::mpl::integral_c< int, Item::value >,
                                                     boost::mpl::integral_c< int, Item::value > > >::type type;
        };

        typedef typename boost::mpl::fold<
            boost::mpl::range_c< uint_t, 0, boost::mpl::size< typename Esf::args_t >::value >,
            boost::mpl::map0<>,
            insert_< boost::mpl::_1, boost::mpl::_2 > >::type trivial_args_map_t;

        typedef icgrid::iterate_domain_remapper< IterateDomain,
            trivial_args_map_t,
            typename Esf::location_type,
            Color::color_t::value > type;
    };

} // namespace gridtools

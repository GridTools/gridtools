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

    namespace strgrid {

        namespace _impl {
            template < typename T >
            struct iterate_domain_remapper_base_iterate_domain;

            template < typename IterateDomain, typename EsfArgsMap, template < typename, typename > class Impl >
            struct iterate_domain_remapper_base_iterate_domain< Impl< IterateDomain, EsfArgsMap > > {
                typedef IterateDomain type;
            };

            template < typename T >
            struct iterate_domain_remapper_base_esf_args_map;

            template < typename IterateDomain, typename EsfArgsMap, template < typename, typename > class Impl >
            struct iterate_domain_remapper_base_esf_args_map< Impl< IterateDomain, EsfArgsMap > > {
                typedef EsfArgsMap type;
            };
        }

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
            typedef typename _impl::iterate_domain_remapper_base_iterate_domain< IterateDomainEvaluatorImpl >::type
                iterate_domain_t;
            static const uint_t N_DATA_POINTERS = iterate_domain_t::N_DATA_POINTERS;

          protected:
            iterate_domain_t &m_iterate_domain;

          public:
            typedef typename _impl::iterate_domain_remapper_base_esf_args_map< IterateDomainEvaluatorImpl >::type
                esf_args_map_t;

            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< iterate_domain_t >::value), "Internal Error: wrong type");
            typedef typename iterate_domain_t::esf_args_t esf_args_t;

#ifdef CXX11_ENABLED
            template < typename Accessor >
            using accessor_return_type = typename iterate_domain_t::template accessor_return_type<
                typename remap_accessor_type< Accessor, esf_args_map_t >::type >;
#else  // CXX11_ENABLED
            template < typename Accessor >
            struct accessor_return_type {
                typedef typename iterate_domain_t::template accessor_return_typ_impl<
                    typename remap_accessor_type< Accessor, esf_args_map_t >::type >::type type;
            };
#endif // CXX11_ENABLED

            GT_FUNCTION
            explicit iterate_domain_remapper_base(iterate_domain_t &iterate_domain)
                : m_iterate_domain(iterate_domain) {}

            GT_FUNCTION
            iterate_domain_t const &get() const { return m_iterate_domain; }

            /** shifting the IDs of the placeholders and forwarding to the iterate_domain () operator*/
            template < typename Accessor >
            GT_FUNCTION
#ifdef CXX11_ENABLED
                auto
                operator()(Accessor const &arg)
                    -> decltype(m_iterate_domain(typename remap_accessor_type< Accessor, esf_args_map_t >::type(arg)))
#else  // CXX11_ENABLED
                typename iterate_domain_t::template accessor_return_type<
                    typename remap_accessor_type< Accessor, esf_args_map_t >::type >::type
                operator()(Accessor const &arg)
#endif // CXX11_ENABLED
            {
                typedef typename remap_accessor_type< Accessor, esf_args_map_t >::type remap_accessor_t;
                const remap_accessor_t tmp_(arg);
                return m_iterate_domain(tmp_);
            }

#ifdef CUDA8 // i.e. CXX11_ENABLED on host
            /** shifting the IDs of the placeholders and forwarding to the iterate_domain () operator*/
            template < typename Accessor, typename... Pairs >
            GT_FUNCTION auto operator()(accessor_mixed< Accessor, Pairs... > const &arg) -> decltype(m_iterate_domain(
                accessor_mixed< typename remap_accessor_type< Accessor, esf_args_map_t >::type, Pairs... >(arg))) {
                typedef accessor_mixed< typename remap_accessor_type< Accessor, esf_args_map_t >::type, Pairs... >
                    remap_accessor_t;
                // const remap_accessor_t tmp_(arg);
                return m_iterate_domain(remap_accessor_t(arg));
            }
#endif // CUDA8

#ifdef CXX11_ENABLED
            /**@brief returns the dimension of the storage corresponding to the given accessor

               Useful to determine the loop bounds, when looping over a dimension from whithin a kernel
               NOTE: shifting the IDs of the placeholders and forwarding to the iterate_domain () operator
            */
            template < ushort_t Coordinate, typename Accessor >
            GT_FUNCTION uint_t get_storage_dim(Accessor acc_) const {
                GRIDTOOLS_STATIC_ASSERT(is_accessor< Accessor >::value, "wrong type");
                typedef typename remap_accessor_type< Accessor, esf_args_map_t >::type remap_accessor_t;
                return m_iterate_domain.get_storage_dim< Coordinate >(remap_accessor_t(acc_));
            }
#endif
        };

        /**
         * @class iterate_domain_remapper
         * default iterate domain remapper when positional information is not required
         * @param IterateDomain iterate domain
         * @param EsfArgsMap map from ESF arguments to iterate domain position of args.
         */
        template < typename IterateDomain, typename EsfArgsMap >
        class iterate_domain_remapper
            : public iterate_domain_remapper_base< iterate_domain_remapper< IterateDomain, EsfArgsMap > > // CRTP
        {
            DISALLOW_COPY_AND_ASSIGN(iterate_domain_remapper);

          public:
            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< IterateDomain >::value), "Internal Error: wrong type");
            typedef iterate_domain_remapper_base< iterate_domain_remapper< IterateDomain, EsfArgsMap > > super;

            GT_FUNCTION
            explicit iterate_domain_remapper(IterateDomain &iterate_domain) : super(iterate_domain) {}
        };

        /**
         * @class positional_iterate__domain_remapper
         * iterate domain remapper when positional information is required
         * @param IterateDomain iterate domain
         * @param EsfArgsMap map from ESF arguments to iterate domain position of args.
         */
        template < typename IterateDomain, typename EsfArgsMap >
        class positional_iterate_domain_remapper
            : public iterate_domain_remapper_base<
                  positional_iterate_domain_remapper< IterateDomain, EsfArgsMap > > // CRTP
        {
            DISALLOW_COPY_AND_ASSIGN(positional_iterate_domain_remapper);

          public:
            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< IterateDomain >::value), "Internal Error: wrong type");
            typedef iterate_domain_remapper_base< positional_iterate_domain_remapper< IterateDomain, EsfArgsMap > >
                super;

            GT_FUNCTION
            explicit positional_iterate_domain_remapper(const IterateDomain &iterate_domain) : super(iterate_domain) {}

            GT_FUNCTION
            uint_t i() const { return this->m_iterate_domain.i(); }

            GT_FUNCTION
            uint_t j() const { return this->m_iterate_domain.j(); }

            GT_FUNCTION
            uint_t k() const { return this->m_iterate_domain.k(); }
        };
    } // namespace strgrid

    /** Metafunction to query an iterate domain if it's positional. Specialization for
        iterate_domain_remapper
    */
    template < typename T, typename U >
    struct is_positional_iterate_domain< strgrid::iterate_domain_remapper< T, U > > : boost::false_type {};

    /** Metafunction to query an iterate domain if it's positional. Specialization for
        positional_iterate_domain_remapper
    */
    template < typename T, typename U >
    struct is_positional_iterate_domain< strgrid::positional_iterate_domain_remapper< T, U > > : boost::true_type {};

    /** Metafunction to query a type is an iterate domain.
    */
    template < typename T, typename U >
    struct is_iterate_domain< strgrid::iterate_domain_remapper< T, U > > : boost::true_type {};

    /** Metafunction to query if a type is an iterate domain.
        positional_iterate_domain_remapper
    */
    template < typename T, typename U >
    struct is_iterate_domain< strgrid::positional_iterate_domain_remapper< T, U > > : boost::true_type {};

    /**
     * @struct get_iterate_domain_remapper
     * metafunction that computes the iterate_domain_remapper from the iterate domain type
     */
    template < typename IterateDomain, typename EsfArgsMap >
    struct get_iterate_domain_remapper {
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< IterateDomain >::value), "Internal Error: wrong type");
        template < typename _IterateDomain, typename _EsfArgsMap >
        struct select_basic_iterate_domain_remapper {
            typedef strgrid::iterate_domain_remapper< _IterateDomain, _EsfArgsMap > type;
        };
        template < typename _IterateDomain, typename _EsfArgsMap >
        struct select_positional_iterate_domain_remapper {
            typedef strgrid::positional_iterate_domain_remapper< _IterateDomain, _EsfArgsMap > type;
        };

        typedef typename boost::mpl::eval_if< is_positional_iterate_domain< IterateDomain >,
            select_positional_iterate_domain_remapper< IterateDomain, EsfArgsMap >,
            select_basic_iterate_domain_remapper< IterateDomain, EsfArgsMap > >::type type;
    };

} // namespace gridtools

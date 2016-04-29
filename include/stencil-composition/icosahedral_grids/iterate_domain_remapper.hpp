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

          protected:
            const iterate_domain_t &m_iterate_domain;

          public:
            typedef typename _impl::iterate_domain_remapper_base_esf_args_map< IterateDomainEvaluatorImpl >::type
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
            explicit iterate_domain_remapper_base(const iterate_domain_t &iterate_domain)
                : m_iterate_domain(iterate_domain) {}

            GT_FUNCTION
            iterate_domain_t const &get() const { return m_iterate_domain; }

            /** shifting the IDs of the placeholders and forwarding to the iterate_domain () operator*/
            template < typename Accessor >
            GT_FUNCTION auto operator()(Accessor const &arg) const
                -> decltype(m_iterate_domain(typename remap_accessor_type< Accessor, esf_args_map_t >::type(arg))) {
                typedef typename remap_accessor_type< Accessor, esf_args_map_t >::type remap_accessor_t;
                return m_iterate_domain(remap_accessor_t(arg));
            }

            template < typename ValueType, typename LocationTypeT, typename Reduction, uint_t I, typename L, int_t R >
            GT_FUNCTION auto operator()(
                on_neighbors_impl< ValueType, LocationTypeT, Reduction, accessor< I, enumtype::in, L, extent< R > > >
                    onneighbors) const
                -> decltype(
                    m_iterate_domain(typename remap_on_neighbors< on_neighbors_impl< ValueType,
                                                                      LocationTypeT,
                                                                      Reduction,
                                                                      accessor< I, enumtype::in, L, extent< R > > >,
                        typename remap_accessor_type< accessor< I, enumtype::in, L, extent< R > >,
                                                                      esf_args_map_t >::type >::type(onneighbors))) {
                typedef on_neighbors_impl< ValueType,
                    LocationTypeT,
                    Reduction,
                    accessor< I, enumtype::in, L, extent< R > > > on_neighbors_t;

                typedef accessor< I, enumtype::in, L, extent< R > > accessor_t;
                typedef typename remap_accessor_type< accessor_t, esf_args_map_t >::type remap_accessor_t;
                typedef typename remap_on_neighbors< on_neighbors_t, remap_accessor_t >::type remap_on_neighbors_t;
                return m_iterate_domain(remap_on_neighbors_t(onneighbors));
            }
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
            explicit iterate_domain_remapper(const IterateDomain &iterate_domain) : super(iterate_domain) {}
        };
    }

    /** Metafunction to query an iterate domain if it's positional. Specialization for
        iterate_domain_remapper
    */
    template < typename T, typename U >
    struct is_positional_iterate_domain< icgrid::iterate_domain_remapper< T, U > > : boost::false_type {};

    /** Metafunction to query a type is an iterate domain.
    */
    template < typename T, typename U >
    struct is_iterate_domain< icgrid::iterate_domain_remapper< T, U > > : boost::true_type {};

    /**
     * @struct get_iterate_domain_remapper
     * metafunction that computes the iterate_domain_remapper from the iterate domain type
     */
    template < typename IterateDomain, typename EsfArgsMap >
    struct get_iterate_domain_remapper {
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< IterateDomain >::value), "Internal Error: wrong type");

        typedef icgrid::iterate_domain_remapper< IterateDomain, EsfArgsMap > type;
    };

} // namespace gridtools

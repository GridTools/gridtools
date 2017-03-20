/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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

            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< iterate_domain_t >::value), GT_INTERNAL_ERROR);
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
            GT_FUNCTION auto operator()(Accessor const &arg) const -> decltype(
                m_iterate_domain(color_t(), typename remap_accessor_type< Accessor, esf_args_map_t >::type(arg))) {
                typedef typename remap_accessor_type< Accessor, esf_args_map_t >::type remap_accessor_t;
                return m_iterate_domain(color_t(), remap_accessor_t(arg));
            }

            /**
             * helper to dereference the value (using an iterate domain) of an accessor
             * (specified with an Index from within a variadic pack of Accessors). It is meant to be used as
             * a functor of a apply_gt_integer_sequence, where the Index is provided from the integer sequence
             * @tparam ValueType value type of the computation
             */
            template < typename ValueType >
            struct it_domain_evaluator {

                /**
                 * @tparam Idx index being processed from within an apply_gt_integer_sequence
                 */
                template < int Idx >
                struct apply_t {

                    GT_FUNCTION
                    constexpr apply_t() {}

                    /**
                     * @tparam Neighbors type locates the position of a neighbor element in the grid. If can be:
                     *     * a quad of values indicating the {i,c,j,k} positions or
                     *     * an integer indicating the absolute index in the storage
                     * @tparam IterateDomain is an iterate domain
                     * @tparam Accessors variadic pack of accessors being processed by the apply_gt_integer_sequence
                     *     and to be evaluated by the iterate domain
                     */
                    template < typename Neighbors, typename IterateDomain, typename... Accessors >
                    GT_FUNCTION static ValueType apply(Neighbors const &__restrict__ neighbors,
                        IterateDomain const &iterate_domain,
                        Accessors &__restrict__... args_) {
                        return iterate_domain._evaluate(get_from_variadic_pack< Idx >::apply(args_...), neighbors);
                    }
                };
            };

            /**
             * returns true if variadic pack is a pack of accessors and the location type of the neighbors is the same
             * as
             * the location type of the ESF.
             */
            template < typename NeighborsLocationType, typename EsfLocationType, typename... Accessors >
            struct accessors_on_same_color_neighbors {
                typedef typename boost::mpl::and_<
                    typename is_sequence_of< typename variadic_to_vector< Accessors... >::type, is_accessor >::type,
                    typename boost::is_same< NeighborsLocationType, EsfLocationType >::type >::type type;
            };

            /**
             * returns true if variadic pack is a pack of accessors and the location type of the neighbors is not the
             * same
             * as
             * the location type of the ESF.
             */
            template < typename NeighborsLocationType, typename EsfLocationType, typename... Accessors >
            struct accessors_on_different_color_neighbors {
                typedef typename boost::mpl::and_<
                    typename is_sequence_of< typename variadic_to_vector< Accessors... >::type, is_accessor >::type,
                    typename is_not_same< NeighborsLocationType, EsfLocationType >::type >::type type;
            };

            /**
             * data structure that holds data needed by the reduce_tuple functor
             * @tparam ValueType value type of the computation
             * @tparam NeighborsArray type locates the position of a neighbor element in the grid. If can be:
             *     * a quad of values indicating the {i,c,j,k} positions or
             *     * an integer indicating the absolute index in the storage
             * @tparam Reduction this is the user lambda specified to expand the on_XXX keyword
             * @tparam IterateDomain is an iterate domain
             */
            template < typename ValueType, typename NeighborsArray, typename Reduction, typename IterateDomain >
            struct reduce_tuple_data_holder {
                Reduction const &m_reduction;
                NeighborsArray const m_neighbors;
                IterateDomain const &m_iterate_domain;
                ValueType &m_result;

              public:
                GT_FUNCTION
                reduce_tuple_data_holder(Reduction const &reduction,
                    NeighborsArray const neighbors,
                    ValueType &result,
                    IterateDomain const &iterate_domain)
                    : m_reduction(reduction), m_neighbors(neighbors), m_result(result),
                      m_iterate_domain(iterate_domain) {}
            };

            /**
             * functor used to expand all the accessors arguments stored in a tuple of a on_neighbors structured.
             * The functor will process all the accessors (i.e. dereference their values of the storages given an
             * neighbors
             * offset)
             * and call the user lambda
             * @tparam ValueType value type of the computation
             * @tparam NeighborsArray type locates the position of a neighbor element in the grid. If can be:
             *     * a quad of values indicating the {i,c,j,k} positions or
             *     * an integer indicating the absolute index in the storage
             * @tparam Reduction this is the user lambda specified to expand the on_XXX keyword
             * @tparam IterateDomain is an iterate domain
             */
            template < typename ValueType, typename NeighborsArray, typename Reduction, typename IterateDomain >
            struct reduce_tuple {

                GRIDTOOLS_STATIC_ASSERT(
                    (boost::is_same<
                         typename boost::remove_const< typename boost::remove_reference< NeighborsArray >::type >::type,
                         unsigned int >::value ||
                        is_array< typename boost::remove_const<
                            typename boost::remove_reference< NeighborsArray >::type >::type >::value),
                    GT_INTERNAL_ERROR);

                GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< IterateDomain >::value), GT_INTERNAL_ERROR);

                typedef reduce_tuple_data_holder< ValueType, NeighborsArray, Reduction, IterateDomain >
                    reduce_tuple_holder_t;

                template < typename... Accessors >
                GT_FUNCTION static void apply(
                    reduce_tuple_holder_t __restrict__ &reducer, Accessors &__restrict__... args) {
                    // we need to call the user functor (Reduction(arg1, arg2, ..., result) )
                    // However we can make here a direct call, since we first need to dereference the address of each
                    // Accessor
                    // given the array with position of the neighbor being accessed (reducer.m_neighbors)
                    // We make use of the apply_gt_integer_sequence in order to operate on each element of the variadic
                    // pack,
                    // dereference its address (it_domain_evaluator) and gather back all the arguments while calling the
                    // user lambda
                    // (Reduction)
                    using seq = apply_gt_integer_sequence<
                        typename make_gt_integer_sequence< int, sizeof...(Accessors) >::type >;

                    reducer.m_result = seq::template apply_lambda< ValueType,
                        Reduction,
                        it_domain_evaluator< ValueType >::template apply_t >(
                        reducer.m_reduction, reducer.m_result, reducer.m_neighbors, reducer.m_iterate_domain, args...);
                }
            };

            // specialization of the () operator for on_neighbors operating on accessors
            // when the location type of the neighbors is the same as the location type of the ESF (iteration space)
            // In this case, dereference of accessors is done using relative offsets instead of absolute indexes
            template < typename ValueType,
                typename SrcColor,
                typename LocationTypeT,
                typename Reduction,
                typename EsfLocationType,
                typename... Accessors >
            GT_FUNCTION typename boost::enable_if<
                typename accessors_on_same_color_neighbors< LocationTypeT, EsfLocationType, Accessors... >::type,
                ValueType >::type
            evaluate(EsfLocationType,
                on_neighbors_impl< ValueType, SrcColor, LocationTypeT, Reduction, Accessors... > onneighbors) const {

                // the neighbors are described as an array of {i,c,j,k} offsets wrt to current position, i.e. an array<
                // array<uint_t, 4>, NumNeighbors>
                constexpr auto neighbors = from< EsfLocationType >::template to< LocationTypeT >::template with_color<
                    static_uint< SrcColor::value > >::offsets();

                // TODO reuse the next code
                ValueType &result = onneighbors.value();

                for (int_t i = 0; i < neighbors.size(); ++i) {

                    typedef decltype(neighbors[i]) neighbors_array_t;
                    reduce_tuple_data_holder< ValueType, neighbors_array_t, Reduction, iterate_domain_t > red(
                        onneighbors.reduction(), neighbors[i], result, m_iterate_domain);
                    // since the on_neighbors store a tuple of accessors (in maps() ), we should explode the
                    // tuple,
                    // so that each element of the tuple is passed as an argument of the user lambda
                    // (which happens in the reduce_tuple).
                    explode< void, reduce_tuple< ValueType, neighbors_array_t, Reduction, iterate_domain_t > >(
                        onneighbors.maps(), red);
                }

                return result;
            }

            // specialization of the () operator for on_neighbors operating on accessors
            template < typename ValueType,
                typename SrcColor,
                typename LocationTypeT,
                typename Reduction,
                typename EsfLocationType,
                typename... Accessors >
            GT_FUNCTION typename boost::enable_if<
                typename accessors_on_different_color_neighbors< LocationTypeT, EsfLocationType, Accessors... >::type,
                ValueType >::type
            evaluate(EsfLocationType,
                on_neighbors_impl< ValueType, SrcColor, LocationTypeT, Reduction, Accessors... > onneighbors) const {

                // the neighbors are described as an array of absolute indices in the storage, i.e. an array<uint?t,
                // NumNeighbors>
                constexpr auto neighbors =
                    connectivity< EsfLocationType, decltype(onneighbors.location()), SrcColor::value >::offsets();

                ValueType &result = onneighbors.value();

                for (int_t i = 0; i < neighbors.size(); ++i) {

                    typedef decltype(neighbors[i]) neighbors_array_t;
                    reduce_tuple_data_holder< ValueType, neighbors_array_t, Reduction, iterate_domain_t > red(
                        onneighbors.reduction(), neighbors[i], result, m_iterate_domain);
                    // since the on_neighbors store a tuple of accessors (in maps() ), we should explode the tuple,
                    // so that each element of the tuple is passed as an argument of the user lambda
                    // (which happens in the reduce_tuple).
                    explode< void, reduce_tuple< ValueType, neighbors_array_t, Reduction, iterate_domain_t > >(
                        onneighbors.maps(), red);
                }

                return result;
            }

            template < typename ValueType, typename LocationTypeT, typename Reduction, typename... Accessors >
            GT_FUNCTION ValueType operator()(
                on_neighbors< ValueType, LocationTypeT, Reduction, Accessors... > onneighbors) const {

                typedef on_neighbors_impl< ValueType,
                    color_t,
                    LocationTypeT,
                    Reduction,
                    typename remap_accessor_type< Accessors, esf_args_map_t >::type... > remap_accessor_t;
                return evaluate(esf_location_type_t(), remap_accessor_t(onneighbors));
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
            GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< IterateDomain >::value), GT_INTERNAL_ERROR);
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
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< IterateDomain >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_location_type< EsfLocationType >::value), GT_INTERNAL_ERROR);

        typedef icgrid::iterate_domain_remapper< IterateDomain, EsfArgsMap, EsfLocationType, Color > type;
    };

    /**
     * @struct get_trivial_iterate_domain_remapper
     * metafunction that computes a trivial iterate_domain_remapper where all the accessors are mapped to themselves
     */
    template < typename IterateDomain, typename Esf, typename Color >
    struct get_trivial_iterate_domain_remapper {
        GRIDTOOLS_STATIC_ASSERT((is_iterate_domain< IterateDomain >::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_esf_descriptor< Esf >::value), GT_INTERNAL_ERROR);

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
